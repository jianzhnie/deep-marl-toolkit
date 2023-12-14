from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from marltoolkit.utils import (LinearDecayScheduler, MultiStepScheduler,
                               check_model_method, hard_target_update)

from .base_agent import BaseAgent


class ACAgent(BaseAgent):
    """ ACAgent algorithm
    Args:
        agent_model (nn.Model): agents' local q network for decision making.
        critic_model (nn.Model): A mixing network which takes local q values as input
            to construct a global Q network.
        double_q (bool): Double-DQN.
        gamma (float): discounted factor for reward computation.
        lr (float): learning rate.
        clip_grad_norm (None, or float): clipped value of gradients' global norm.
    """

    def __init__(self,
                 agent_model: nn.Module = None,
                 critic_model: nn.Module = None,
                 n_agents: int = None,
                 double_q: bool = True,
                 q_nstep: int = 5,
                 total_steps: int = 1e6,
                 gamma: float = 0.99,
                 entropy_coef: float = 0.01,
                 learning_rate: float = 0.0005,
                 min_learning_rate: float = 0.0001,
                 exploration_start: float = 1.0,
                 min_exploration: float = 0.01,
                 add_value_last_step: bool = False,
                 update_target_interval: int = 100,
                 update_learner_freq: int = 1,
                 clip_grad_norm: float = 10,
                 device: str = 'cpu'):

        check_model_method(agent_model, 'init_hidden', self.__class__.__name__)
        check_model_method(agent_model, 'forward', self.__class__.__name__)
        assert isinstance(gamma, float)
        assert isinstance(learning_rate, float)

        self.n_agents = n_agents
        self.q_nstep = q_nstep
        self.double_q = double_q
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.clip_grad_norm = clip_grad_norm
        self.global_steps = 0
        self.exploration = exploration_start
        self.min_exploration = min_exploration
        self.add_value_last_step = add_value_last_step
        self.target_update_count = 0
        self.update_target_interval = update_target_interval
        self.update_learner_freq = update_learner_freq

        self.device = device
        self.agent_model = agent_model
        self.target_agent_model = deepcopy(self.agent_model)
        self.agent_model.to(device)
        self.target_agent_model.to(device)

        self.agent_params = list(self.agent_model.parameters())
        self.agent_optimiser = torch.optim.Adam(
            params=self.agent_params, lr=learning_rate)

        self.critic_model = critic_model
        self.target_critic_model = deepcopy(self.critic_model)
        self.critic_model.to(device)
        self.target_critic_model.to(device)
        self.critic_params = list(self.critic_model.parameters())

        self.critic_optimiser = torch.optim.Adam(
            params=self.critic_params, lr=learning_rate)

        self.ep_scheduler = LinearDecayScheduler(exploration_start,
                                                 total_steps * 0.8)

        lr_steps = [total_steps * 0.5, total_steps * 0.8]
        self.lr_scheduler = MultiStepScheduler(
            start_value=learning_rate,
            max_steps=total_steps,
            milestones=lr_steps,
            decay_factor=0.5)

    def reset_agent(self, batch_size=1):
        self._init_hidden_states(batch_size)

    def _init_hidden_states(self, batch_size):
        self.hidden_states = self.agent_model.init_hidden()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(
                batch_size, self.n_agents, -1)

        self.target_hidden_states = self.target_agent_model.init_hidden()
        if self.target_hidden_states is not None:
            self.target_hidden_states = self.target_hidden_states.unsqueeze(
                0).expand(batch_size, self.n_agents, -1)

    def sample(self, obs, available_actions):
        ''' sample actions via epsilon-greedy
        Args:
            obs (np.ndarray):               (n_agents, obs_shape)
            available_actions (np.ndarray): (n_agents, n_actions)
        Returns:
            actions (np.ndarray): sampled actions of agents
        '''
        epsilon = np.random.random()
        if epsilon < self.exploration:
            available_actions = torch.tensor(
                available_actions, dtype=torch.float32)
            actions_dist = Categorical(available_actions)
            actions = actions_dist.sample().long().cpu().detach().numpy()

        else:
            actions = self.predict(obs, available_actions)

        # update exploration
        self.exploration = max(self.ep_scheduler.step(), self.min_exploration)
        return actions

    def predict(self, obs, available_actions):
        '''take greedy actions
        Args:
            obs (np.ndarray):               (n_agents, obs_shape)
            available_actions (np.ndarray): (n_agents, n_actions)
        Returns:
            actions (np.ndarray):           (n_agents, )
        '''
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        available_actions = torch.tensor(
            available_actions, dtype=torch.long, device=self.device)
        agents_q, self.hidden_states = self.agent_model(
            obs, self.hidden_states)
        # mask unavailable actions
        agents_q[available_actions == 0] = -1e10
        actions = agents_q.max(dim=1)[1].detach().cpu().numpy()
        return actions

    def update_target(self):
        hard_target_update(self.agent_model, self.target_agent_model)
        hard_target_update(self.critic_model, self.target_critic_model)

    def learn(self, state_batch, actions_batch, reward_batch, terminated_batch,
              obs_batch, available_actions_batch, filled_batch, **kwargs):
        '''
        Args:
            state (np.ndarray):                   (batch_size, T, state_shape)
            actions (np.ndarray):                 (batch_size, T, n_agents)
            reward (np.ndarray):                  (batch_size, T, 1)
            terminated (np.ndarray):              (batch_size, T, 1)
            obs (np.ndarray):                     (batch_size, T, n_agents, obs_shape)
            available_actions_batch (np.ndarray): (batch_size, T, n_agents, n_actions)
            filled_batch (np.ndarray):            (batch_size, T, 1)
        Returns:
            mean_loss (float): train loss
            mean_td_error (float): train TD error
        '''
        # update target model
        if self.global_steps % self.update_target_interval == 0:
            self.update_target()
            self.target_update_count += 1

        self.global_steps += 1

        # set the actions to torch.Long
        actions_batch = actions_batch.to(self.device, dtype=torch.long)
        # get the batch_size and episode_length
        batch_size = state_batch.shape[0]
        episode_len = state_batch.shape[1]

        # get the relevant quantitles
        reward_batch = reward_batch[:, :-1, :]
        actions_batch = actions_batch[:, :-1, :].unsqueeze(-1)
        terminated_batch = terminated_batch[:, :-1, :]
        filled_batch = filled_batch[:, :-1, :]

        mask = (1 - filled_batch) * (1 - terminated_batch)
        critic_mask = mask.clone()
        # Calculate estimated Q-Values
        local_qs = []
        target_local_qs = []
        self._init_hidden_states(batch_size)
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            # obs: (batch_size * n_agents, obs_shape)
            obs = obs.reshape(-1, obs_batch.shape[-1])
            # Calculate estimated Q-Values
            local_q, self.hidden_states = self.agent_model(
                obs, self.hidden_states)
            #  local_q: (batch_size * n_agents, n_actions) -->  (batch_size, n_agents, n_actions)
            local_q = local_q.reshape(batch_size, self.n_agents, -1)
            local_qs.append(local_q)

            # Calculate the Q-Values necessary for the target
            target_local_q, self.target_hidden_states = self.target_agent_model(
                obs, self.target_hidden_states)
            # target_local_q: (batch_size * n_agents, n_actions) -->  (batch_size, n_agents, n_actions)
            target_local_q = target_local_q.view(batch_size, self.n_agents, -1)
            target_local_qs.append(target_local_q)

        # Concat over time
        local_qs = torch.stack(local_qs, dim=1)
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_local_qs = torch.stack(target_local_qs[1:], dim=1)

        advantages, mean_td_error, critic_loss = self.train_critic_sequential(
            obs_batch, reward_batch, critic_mask)
        advantages = advantages.detach()
        # Calculate policy grad with mask
        pi = local_qs
        pi[mask == 0] = 1.0
        pi_taken = torch.gather(pi, dim=3, index=actions_batch).squeeze(3)
        log_pi_taken = torch.log(pi_taken + 1e-10)

        entropy = -torch.sum(pi * torch.log(pi + 1e-10), dim=-1)
        pg_loss = -((advantages * log_pi_taken + self.entropy_coef * entropy) *
                    mask).sum() / mask.sum()

        # Optimise agents
        self.agent_optimiser.zero_grad()
        pg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent_params,
                                       self.args.grad_norm_clip)
        self.agent_optimiser.step()

        return pg_loss.item(), critic_loss.item(), mean_td_error.item()

    def train_critic_sequential(self, obs_batch, rewards, mask):
        with torch.no_grad():
            target_vals = self.target_critic_model(obs_batch)
            target_vals = target_vals.squeeze(3)

        target_returns = self.nstep_returns(rewards, mask, target_vals,
                                            self.q_nstep)

        v = self.critic_model(obs_batch)[:, :-1].squeeze(3)
        td_error = (target_returns.detach() - v)
        masked_td_error = td_error * mask
        mean_td_error = masked_td_error.sum() / mask.sum()
        critic_loss = (masked_td_error**2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_params, self.clip_grad_norm)
        self.critic_optimiser.step()
        return masked_td_error, mean_td_error, critic_loss

    def nstep_returns(self, rewards, mask, values, nsteps):
        nstep_values = torch.zeros_like(values[:, :-1])
        for t_start in range(rewards.size(1)):
            nstep_return_t = torch.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += self.gamma**step * values[:, t] * mask[:,
                                                                             t]
                elif t == rewards.size(1) - 1 and self.add_value_last_step:
                    nstep_return_t += self.gamma**step * rewards[:,
                                                                 t] * mask[:,
                                                                           t]
                    nstep_return_t += self.gamma**(step + 1) * values[:, t + 1]
                else:
                    nstep_return_t += self.gamma**step * rewards[:,
                                                                 t] * mask[:,
                                                                           t]
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values
