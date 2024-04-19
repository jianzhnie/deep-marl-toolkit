import os
from copy import deepcopy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from marltoolkit.agents.base_agent import BaseAgent
from marltoolkit.utils import (LinearDecayScheduler, MultiStepScheduler,
                               hard_target_update)


class QMixAgent(BaseAgent):
    """QMIX algorithm
    Args:
        actor_model (nn.Model): agents' local q network for decision making.
        mixer_model (nn.Model): A mixing network which takes local q values as input
            to construct a global Q network.
        double_q (bool): Double-DQN.
        gamma (float): discounted factor for reward computation.
        lr (float): learning rate.
        clip_grad_norm (None, or float): clipped value of gradients' global norm.
    """

    def __init__(
        self,
        actor_model: nn.Module = None,
        mixer_model: nn.Module = None,
        num_envs: int = 1,
        num_agents: int = None,
        double_q: bool = True,
        total_steps: int = 1e6,
        gamma: float = 0.99,
        optimizer_type: str = 'rmsprop',
        learning_rate: float = 0.0005,
        min_learning_rate: float = 0.0001,
        egreedy_exploration: float = 1.0,
        min_exploration: float = 0.01,
        target_update_interval: int = 100,
        learner_update_freq: int = 1,
        clip_grad_norm: float = 10,
        optim_alpha: float = 0.99,
        optim_eps: float = 0.00001,
        device: str = 'cpu',
    ) -> None:
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.double_q = double_q
        self.gamma = gamma
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.clip_grad_norm = clip_grad_norm
        self.global_steps = 0
        self.exploration = egreedy_exploration
        self.min_exploration = min_exploration
        self.target_update_count = 0
        self.target_update_interval = target_update_interval
        self.learner_update_freq = learner_update_freq

        self.device = device
        self.actor_model = actor_model.to(device)
        self.target_actor_model = deepcopy(self.actor_model).to(device)
        self.params = list(self.actor_model.parameters())

        self.mixer_model = mixer_model.to(device)
        self.target_mixer_model = deepcopy(self.mixer_model).to(device)
        self.params += list(self.mixer_model.parameters())

        if self.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(params=self.params,
                                              lr=self.learning_rate)
        else:
            self.optimizer = torch.optim.RMSprop(
                params=self.params,
                lr=self.learning_rate,
                alpha=optim_alpha,
                eps=optim_eps,
            )

        self.ep_scheduler = LinearDecayScheduler(egreedy_exploration,
                                                 total_steps * 0.8)

        lr_milstons = [total_steps * 0.5, total_steps * 0.8]
        self.lr_scheduler = MultiStepScheduler(
            start_value=learning_rate,
            max_steps=total_steps,
            milestones=lr_milstons,
            decay_factor=0.1,
        )

        # 执行过程中，要为每个agent都维护一个 hidden_state
        # 学习过程中，要为每个agent都维护一个 hidden_state、target_hidden_state
        self.hidden_state = None
        self.target_hidden_state = None

    def init_hidden_states(self, batch_size: int = 1) -> None:
        """Initialize hidden states for each agent.

        Args:
            batch_size (int): batch size
        """
        self.hidden_state = self.actor_model.init_hidden(batch_size *
                                                         self.num_agents)
        self.target_hidden_state = self.target_actor_model.init_hidden(
            batch_size * self.num_agents)

    def sample(self, obs: torch.Tensor, available_actions: torch.Tensor):
        """sample actions via epsilon-greedy
        Args:
            obs (np.ndarray):               (num_agents, obs_shape)
            available_actions (np.ndarray): (num_agents, n_actions)
        Returns:
            actions (np.ndarray): sampled actions of agents
        """
        epsilon = np.random.random()
        if epsilon < self.exploration:
            available_actions = torch.tensor(available_actions)
            actions_dist = Categorical(available_actions)
            actions = actions_dist.sample().numpy()

        else:
            actions = self.predict(obs, available_actions)

        # update exploration
        self.exploration = max(self.ep_scheduler.step(), self.min_exploration)
        return actions

    def predict(self, obs: torch.Tensor, available_actions: torch.Tensor):
        """take greedy actions
        Args:
            obs (np.ndarray):               (num_agents, obs_shape)
            available_actions (np.ndarray): (num_agents, n_actions)
        Returns:
            actions (np.ndarray):           (num_agents, )
        """
        obs = torch.tensor(obs, dtype=torch.float32,
                           device=self.device).unsqueeze(1)
        available_actions = torch.tensor(available_actions,
                                         dtype=torch.long,
                                         device=self.device)
        agents_q, self.hidden_state = self.actor_model(obs, self.hidden_state)
        available_actions = available_actions.unsqueeze(1)
        # mask unavailable actions
        agents_q[available_actions == 0] = -1e10
        actions = agents_q.max(dim=2)[1].detach().cpu().numpy()
        return actions

    def update_target(self) -> None:
        hard_target_update(self.actor_model, self.target_actor_model)
        if self.mixer_model is not None:
            hard_target_update(self.mixer_model, self.target_mixer_model)

    def learn(self, episode_data: Dict[str, np.ndarray]):
        """Update the model from a batch of experiences.

        Args: episode_data (dict) with the following:

            - obs (np.ndarray):                     (batch_size, T, num_agents, obs_shape)
            - state (np.ndarray):                   (batch_size, T, state_shape)
            - actions (np.ndarray):                 (batch_size, T, num_agents)
            - rewards (np.ndarray):                 (batch_size, T, 1)
            - dones (np.ndarray):                   (batch_size, T, 1)
            - available_actions (np.ndarray):        (batch_size, T, num_agents, n_actions)
            - filled (np.ndarray):                  (batch_size, T, 1)

        Returns:
            - mean_loss (float): train loss
            - mean_td_error (float): train TD error
        """
        # get the data from episode_data buffer
        obs_episode = episode_data['obs']
        state_episode = episode_data['state']
        actions_episode = episode_data['actions']
        available_actions_episode = episode_data['available_actions']
        rewards_episode = episode_data['rewards']
        dones_episode = episode_data['dones']
        filled_episode = episode_data['filled']

        # update target model
        if self.global_steps % self.target_update_interval == 0:
            self.update_target()
            self.target_update_count += 1

        self.global_steps += 1

        # set the actions to torch.Long
        actions_episode = torch.tensor(actions_episode,
                                       dtype=torch.long,
                                       device=self.device)
        # get the batch_size and episode_length
        (batch_size, episode_len, num_agents, obs_dim) = obs_episode.shape
        n_actions = available_actions_episode.shape[-1]

        # get the relevant quantitles
        dones_episode = dones_episode.float()
        filled_episode = filled_episode.float()
        actions_episode = actions_episode.unsqueeze(-1)

        mask = (1 - dones_episode) * (1 - filled_episode)

        # Calculate estimated Q-Values
        obs_episode = obs_episode.reshape(batch_size * num_agents, episode_len,
                                          obs_dim)
        local_qs, self.hidden_state = self.actor_model(obs_episode,
                                                       self.hidden_state)
        target_local_qs, self.target_hidden_state = self.target_actor_model(
            obs_episode, self.target_hidden_state)

        # Concat over time
        local_qs = local_qs.reshape(batch_size, episode_len, num_agents,
                                    n_actions)
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_local_qs = target_local_qs.reshape(batch_size, episode_len,
                                                  num_agents, n_actions)
        # Pick the Q-Values for the actions taken by each agent
        # Remove the last dim
        chosen_action_local_qs = torch.gather(local_qs,
                                              dim=3,
                                              index=actions_episode)
        # mask unavailable actions
        target_local_qs[available_actions_episode == 0] = -1e10

        # Max over target Q-Values
        if self.double_q:
            # Get actions that maximise live Q (for double q-learning)
            local_qs_detach = local_qs.clone().detach()
            local_qs_detach[available_actions_episode == 0] = -1e10
            cur_max_actions = local_qs_detach.max(dim=3, keepdim=True)[1]
            target_local_max_qs = torch.gather(target_local_qs,
                                               dim=3,
                                               index=cur_max_actions)
        else:
            # idx0: value, idx1: index
            target_local_max_qs = target_local_qs.max(dim=3)[0]

        # Mixing network
        # mix_net, input: ([Q1, Q2, ...], state), output: Q_total
        if self.mixer_model is not None:
            chosen_action_global_qs = self.mixer_model(chosen_action_local_qs,
                                                       state_episode)
            target_global_max_qs = self.target_mixer_model(
                target_local_max_qs, state_episode)

        if self.mixer_model is None:
            target_max_qvals = target_local_max_qs
            chosen_action_qvals = chosen_action_local_qs
        else:
            target_max_qvals = target_global_max_qs
            chosen_action_qvals = chosen_action_global_qs

        # Calculate 1-step Q-Learning targets
        target = rewards_episode + self.gamma * (
            1 - dones_episode) * target_max_qvals
        #  Td-error
        td_error = target.detach() - chosen_action_qvals

        # 0-out the targets that came from padded data
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

        results = {
            'loss': loss.item(),
            'mean_td_error': mean_td_error.item(),
        }
        return results

    def save_model(
        self,
        save_dir: str = None,
        actor_model_name: str = 'actor_model.th',
        mixer_model_name: str = 'mixer_model.th',
        opt_name: str = 'optimizer.th',
    ):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        actor_model_path = os.path.join(save_dir, actor_model_name)
        mixer_model_path = os.path.join(save_dir, mixer_model_name)
        optimizer_path = os.path.join(save_dir, opt_name)
        torch.save(self.actor_model.state_dict(), actor_model_path)
        torch.save(self.mixer_model.state_dict(), mixer_model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)
        print('save model successfully!')

    def load_model(
        self,
        save_dir: str = None,
        actor_model_name: str = 'actor_model.th',
        mixer_model_name: str = 'mixer_model.th',
        opt_name: str = 'optimizer.th',
    ):
        actor_model_path = os.path.join(save_dir, actor_model_name)
        mixer_model_path = os.path.join(save_dir, mixer_model_name)
        optimizer_path = os.path.join(save_dir, opt_name)
        self.actor_model.load_state_dict(torch.load(actor_model_path))
        self.mixer_model.load_state_dict(torch.load(mixer_model_path))
        self.optimizer.load_state_dict(torch.load(optimizer_path))
        print('restore model successfully!')
