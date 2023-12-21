import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from marltoolkit.utils import (LinearDecayScheduler, MultiStepScheduler,
                               check_model_method, hard_target_update)


class QTranAgent(object):
    """ Qtran algorithm
    Args:
        agent_model (nn.Model): agents' local q network for decision making.
        mixer_model (nn.Model): A mixing network which takes local q values as input
            to construct a global Q network.
        double_q (bool): Double-DQN.
        gamma (float): discounted factor for reward computation.
        lr (float): learning rate.
        clip_grad_norm (None, or float): clipped value of gradients' global norm.
    """

    def __init__(self,
                 agent_model: nn.Module = None,
                 mixer_model: nn.Module = None,
                 n_agents: int = None,
                 n_actions: int = None,
                 double_q: bool = True,
                 total_steps: int = 1e6,
                 gamma: float = 0.99,
                 learning_rate: float = 0.0005,
                 min_learning_rate: float = 0.00001,
                 exploration_start: float = 1.0,
                 min_exploration: float = 0.01,
                 update_target_interval: int = 100,
                 update_learner_freq: int = 1,
                 clip_grad_norm: float = 10,
                 optim_alpha: float = 0.99,
                 optim_eps: float = 0.00001,
                 opt_loss_coef: float = 0.1,
                 nopt_min_loss_coef: float = 0.1,
                 device: str = 'cpu'):

        check_model_method(agent_model, 'init_hidden', self.__class__.__name__)
        check_model_method(agent_model, 'forward', self.__class__.__name__)
        check_model_method(mixer_model, 'forward', self.__class__.__name__)
        assert isinstance(gamma, float)
        assert isinstance(learning_rate, float)

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.double_q = double_q
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.clip_grad_norm = clip_grad_norm
        self.global_steps = 0
        self.exploration = exploration_start
        self.min_exploration = min_exploration
        self.target_update_count = 0
        self.update_target_interval = update_target_interval
        self.update_learner_freq = update_learner_freq
        self.opt_loss_coef = opt_loss_coef
        self.nopt_min_loss_coef = nopt_min_loss_coef

        self.device = device
        self.agent_model = agent_model
        self.mixer_model = mixer_model
        self.target_agent_model = deepcopy(self.agent_model)
        self.target_mixer_model = deepcopy(self.mixer_model)
        self.agent_model.to(device)
        self.target_agent_model.to(device)
        self.mixer_model.to(device)
        self.target_mixer_model.to(device)

        self.params = list(self.agent_model.parameters())
        self.params += list(self.mixer_model.parameters())
        self.optimizer = torch.optim.RMSprop(
            params=self.params,
            lr=self.learning_rate,
            alpha=optim_alpha,
            eps=optim_eps)

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
        hard_target_update(self.mixer_model, self.target_mixer_model)

    def learn(self, state_batch, actions_batch, actions_onehot_batch,
              reward_batch, terminated_batch, obs_batch,
              available_actions_batch, filled_batch, **kwargs):
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
        actions_onehot_batch = actions_onehot_batch.to(
            self.device, dtype=torch.long)

        # get the batch_size and episode_length
        batch_size = state_batch.shape[0]
        episode_len = state_batch.shape[1]

        # get the relevant quantitles
        reward_batch = reward_batch[:, :-1, :]
        actions_batch = actions_batch[:, :-1, :].unsqueeze(-1)
        actions_onehot_batch = actions_onehot_batch[:, :-1]
        terminated_batch = terminated_batch[:, :-1, :]
        filled_batch = filled_batch[:, :-1, :]

        mask = (1 - filled_batch) * (1 - terminated_batch)

        # Calculate estimated Q-Values
        local_qs = []
        target_local_qs = []
        local_hidden_states = []
        target_local_hidden_states = []
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
            # hidden_states: (batch_size * n_agents, rnn_hidden_dim)
            # local_hidden = self.hidden_states.reshape(batch_size,
            #                                           self.n_agents, -1)
            local_hidden_states.append(self.hidden_states)

            # Calculate the Q-Values necessary for the target
            target_local_q, self.target_hidden_states = self.target_agent_model(
                obs, self.target_hidden_states)
            # target_local_q: (batch_size * n_agents, n_actions) -->  (batch_size, n_agents, n_actions)
            target_local_q = target_local_q.view(batch_size, self.n_agents, -1)
            target_local_qs.append(target_local_q)
            # traget_local_hidden = self.target_hidden_states.reshape(
            #     batch_size, self.n_agents, -1)
            target_local_hidden_states.append(self.target_hidden_states)

        # local_q 和 target_local_q 是一个列表，列表里装着 episode_len 个数组，
        # 数组的的维度是 (batch_size, n_agents，n_actions).
        # 通过torch.stack() 把该列表转化成 (batch_size, episode_len， n_agents，n_actions)的数组
        # Concat over time
        local_qs = torch.stack(local_qs, dim=1)
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_local_qs = torch.stack(target_local_qs[1:], dim=1)
        # Concat over time
        local_hidden_states = torch.stack(local_hidden_states, dim=1)
        local_hidden_states = local_hidden_states.reshape(
            batch_size, self.n_agents, episode_len, -1).transpose(1, 2)  # btav

        target_local_hidden_states = torch.stack(
            target_local_hidden_states, dim=1)
        target_local_hidden_states = target_local_hidden_states.reshape(
            batch_size, self.n_agents, episode_len, -1).transpose(1, 2)  # btav

        # Pick the Q-Values for the actions taken by each agent
        # Remove the last dim
        chosen_action_local_qs = torch.gather(
            local_qs[:, :-1, :, :], dim=3, index=actions_batch).squeeze(3)

        # mask unavailable actions
        local_qs_detach = local_qs.clone().detach()
        local_qs_detach[available_actions_batch == 0] = -1e10
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_local_qs[available_actions_batch[:, 1:, :] == 0] = -1e10

        # Best joint action computed by target agents
        target_local_max_qs, target_max_actions = target_local_qs.max(
            dim=3, keepdim=True)
        # Best joint-action computed by regular agents
        max_actions_qvals, max_actions_current = local_qs_detach.max(
            dim=3, keepdim=True)

        # Need to argmax across the target agents' actions to compute target joint-action Q-Values
        # Max over target Q-Values
        if self.double_q:
            # Get actions that maximise live Q (for double q-learning)
            max_actions_current_ = torch.zeros(
                size=(batch_size, episode_len, self.n_agents, self.n_actions),
                device=self.device)
            max_actions_current_onehot = max_actions_current_.scatter(
                3, max_actions_current, 1)
            max_actions_onehot = max_actions_current_onehot
        else:
            max_actions = torch.zeros(
                size=(batch_size, episode_len, self.n_agents, self.n_actions),
                device=self.device)
            max_actions_onehot = max_actions.scatter(3, target_max_actions, 1)

        # Joint-action Q-Value estimates
        joint_qs, vs = self.mixer_model(
            state=state_batch[:, :-1],
            hidden_states=local_hidden_states[:, :-1],
            actions=actions_onehot_batch)

        # target Joint-action Q-Value estimates
        target_joint_qs, target_vs = self.target_mixer_model(
            state=state_batch[:, :-1],
            hidden_states=target_local_hidden_states[:, 1:],
            actions=max_actions_onehot[:, 1:])

        # Td loss targets
        td_targets = reward_batch.reshape(-1, 1) + self.gamma * (
            1 - terminated_batch.reshape(-1, 1)) * target_joint_qs
        #  Td-error
        td_error = (joint_qs - td_targets.detach())
        masked_td_error = td_error * mask.reshape(-1, 1)
        td_loss = (masked_td_error**2).sum() / mask.sum()
        # -- TD Loss --

        # -- Opt Loss --
        # Argmax across the current agents' actions
        if not self.double_q:  # Already computed if we're doing double Q-Learning
            max_actions_current_ = torch.zeros(
                size=(batch_size, episode_len, self.n_agents, self.n_actions),
                device=self.device)
            max_actions_current_onehot = max_actions_current_.scatter(
                3, max_actions_current, 1)

        # Don't use the target network and target agent max actions as per author's email
        # Joint-action Q-Value estimates
        max_joint_qs, _ = self.mixer_model(
            state_batch[:, :-1],
            local_hidden_states[:, :-1],
            actions=max_actions_current_onehot[:, :-1])

        # max_actions_qvals = th.gather(mac_out[:, :-1], dim=3, index=max_actions_current[:,:-1])
        opt_error = max_actions_qvals[:, :-1].sum(dim=2).reshape(
            -1, 1) - max_joint_qs.detach() + vs
        masked_opt_error = opt_error * mask.reshape(-1, 1)
        opt_loss = (masked_opt_error**2).sum() / mask.sum()
        # -- Opt Loss --

        # -- Nopt Loss --
        # target_joint_qs, _ = self.target_mixer(batch[:, :-1])
        nopt_values = chosen_action_local_qs.sum(dim=2).reshape(
            -1, 1) - joint_qs.detach() + vs
        # Don't use target networks here either
        nopt_error = nopt_values.clamp(max=0)
        masked_nopt_error = nopt_error * mask.reshape(-1, 1)
        nopt_loss = (masked_nopt_error**2).sum() / mask.sum()
        # -- Nopt loss --

        loss = td_loss + self.opt_loss_coef * opt_loss + self.nopt_min_loss_coef * nopt_loss

        # Optimise
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip_grad_norm)
        self.optimizer.step()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

        return loss.item(), td_loss.item(), opt_loss.item(), nopt_loss.item()

    def save(self,
             save_dir: str = None,
             agent_model_name: str = 'agent_model.th',
             mixer_model_name: str = 'mixer_model.th',
             opt_name: str = 'optimizer.th'):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        agent_model_path = os.path.join(save_dir, agent_model_name)
        mixer_model_path = os.path.join(save_dir, mixer_model_name)
        optimizer_path = os.path.join(save_dir, opt_name)
        torch.save(self.agent_model.state_dict(), agent_model_path)
        torch.save(self.mixer_model.state_dict(), mixer_model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)
        print('save model successfully!')

    def restore(self,
                save_dir: str = None,
                agent_model_name: str = 'agent_model.th',
                mixer_model_name: str = 'mixer_model.th',
                opt_name: str = 'optimizer.th'):
        agent_model_path = os.path.join(save_dir, agent_model_name)
        mixer_model_path = os.path.join(save_dir, mixer_model_name)
        optimizer_path = os.path.join(save_dir, opt_name)
        self.agent_model.load_state_dict(torch.load(agent_model_path))
        self.mixer_model.load_state_dict(torch.load(mixer_model_path))
        self.optimizer.load_state_dict(torch.load(optimizer_path))
        print('restore model successfully!')
