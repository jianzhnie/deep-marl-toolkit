import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from rltoolkit.models.utils import check_model_method, hard_target_update
from rltoolkit.utils.scheduler import LinearDecayScheduler, MultiStepScheduler


class QMixAgent(object):
    """ QMIX algorithm
    Args:
        agent_model (rltoolkit.Model): agents' local q network for decision making.
        mixer_model (rltoolkit.Model): A mixing network which takes local q values as input
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
                 double_q: bool = True,
                 total_steps: int = 1e6,
                 gamma: float = 0.99,
                 learning_rate: float = 0.0005,
                 min_learning_rate: float = 0.0001,
                 exploration_start: float = 1.0,
                 min_exploration: float = 0.01,
                 update_target_interval: int = 100,
                 update_learner_freq: int = 1,
                 clip_grad_norm: float = 10,
                 optim_alpha: float = 0.99,
                 optim_eps: float = 0.00001,
                 device: str = 'cpu'):

        check_model_method(agent_model, 'init_hidden', self.__class__.__name__)
        check_model_method(agent_model, 'forward', self.__class__.__name__)
        check_model_method(mixer_model, 'forward', self.__class__.__name__)
        assert hasattr(mixer_model, 'n_agents') and not callable(
            getattr(mixer_model, 'n_agents',
                    None)), 'mixer_model needs to have attribute n_agents'
        assert isinstance(gamma, float)
        assert isinstance(learning_rate, float)

        self.n_agents = n_agents
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

        # Pick the Q-Values for the actions taken by each agent
        # Remove the last dim
        chosen_action_local_qs = torch.gather(
            local_qs[:, :-1, :, :], dim=3, index=actions_batch).squeeze(3)

        # mask unavailable actions
        target_local_qs[available_actions_batch[:, 1:, :] == 0] = -1e10

        # Max over target Q-Values
        if self.double_q:
            # Get actions that maximise live Q (for double q-learning)
            local_qs_detach = local_qs.clone().detach()
            local_qs_detach[available_actions_batch == 0] = -1e10
            cur_max_actions = local_qs_detach[:, 1:].max(
                dim=3, keepdim=True)[1]
            target_local_max_qs = torch.gather(
                target_local_qs, dim=3, index=cur_max_actions).squeeze(3)
        else:
            # idx0: value, idx1: index
            target_local_max_qs = target_local_qs.max(dim=3)[0]

        # Mixing network
        # mix_net, input: ([Q1, Q2, ...], state), output: Q_total
        if self.mixer_model is not None:
            chosen_action_global_qs = self.mixer_model(chosen_action_local_qs,
                                                       state_batch[:, :-1, :])
            target_global_max_qs = self.target_mixer_model(
                target_local_max_qs, state_batch[:, 1:, :])

        # Calculate 1-step Q-Learning targets
        target = reward_batch + self.gamma * (
            1 - terminated_batch) * target_global_max_qs
        #  Td-error
        td_error = target.detach() - chosen_action_global_qs
        #  0-out the targets that came from padded data
        masked_td_error = td_error * mask
        mean_td_error = masked_td_error.sum() / mask.sum()
        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error**2).sum() / mask.sum()

        # Optimise
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip_grad_norm)
        self.optimizer.step()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

        return loss.item(), mean_td_error.item()

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
