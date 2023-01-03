import os
import sys
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from rltoolkit.models.utils import check_model_method, hard_target_update
from rltoolkit.utils.scheduler import LinearDecayScheduler


class BaseAgent(object):
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
                 learning_rate: float = 0.001,
                 min_learning_rate: float = 0.00001,
                 exploration_start: float = 1.0,
                 min_exploration: float = 0.01,
                 update_target_interval: int = 1000,
                 update_learner_freq: int = 5,
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
        self.global_step = 0
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
        if epsilon > self.exploration:
            actions = self.predict(obs, available_actions)
        else:
            available_actions = torch.tensor(
                available_actions, dtype=torch.float32)
            actions_dist = Categorical(available_actions)
            actions = actions_dist.sample().long().cpu().detach().numpy()

        self.exploration = max(
            self.ep_scheduler.step(self.update_learner_freq),
            self.min_exploration,
        )
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

    def learn(self):
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
        return NotImplemented

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
