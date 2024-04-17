import argparse
from typing import Tuple, Union

import numpy as np

from .base_buffer import BaseBuffer


class SharedReplayBuffer(BaseBuffer):
    """Buffer to store training data.

    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    :param obs_shape: (gym.Space) observation space of agents.
    :param state_shape: (gym.Space) centralized observation space of agents.
    :param action_shape: (gym.Space) action space for agents.
    """

    def __init__(
        self,
        num_envs: int,
        num_agents: int,
        episode_limit: int,
        obs_shape: Union[int, Tuple],
        state_shape: Union[int, Tuple],
        action_shape: Union[int, Tuple],
        reward_shape: Union[int, Tuple],
        done_shape: Union[int, Tuple],
        args: argparse.Namespace,
    ) -> None:
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.episode_limit = episode_limit

        self.hidden_size = args.hidden_size
        self.rnn_layers = args.rnn_layers
        self.gamma = args.gamma
        self.use_gae = args.use_gae
        self.gae_lambda = args.gae_lambda
        self.use_popart = args.use_popart
        self.use_valuenorm = args.use_valuenorm
        self.use_proper_time_limits = args.use_proper_time_limits
        self.algo = args.algorithm_name

        self.obs = np.zeros(
            (self.episode_limit + 1, self.num_envs, *obs_shape),
            dtype=np.float32,
        )
        self.state = np.zeros(
            (self.episode_limit + 1, self.num_envs, *state_shape),
            dtype=np.float32,
        )
        self.actions = np.zeros(
            (self.episode_limit, self.num_envs, num_agents),
            dtype=np.float32,
        )
        self.rnn_hidden_states = np.zeros(
            (
                self.episode_limit + 1,
                self.num_envs,
                self.num_agents,
                self.rnn_layers,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        self.rnn_hidden_states_critic = np.zeros_like(self.rnn_hidden_states)

        self.rewards = np.zeros(
            (self.episode_limit, self.num_envs, 1),
            dtype=np.float32,
        )
        self.value_preds = np.zeros(
            (self.episode_limit + 1, self.num_envs, 1),
            dtype=np.float32,
        )
        self.returns = np.zeros_like(self.value_preds)
        self.advantages = np.zeros(
            (self.episode_limit, self.num_envs, 1),
            dtype=np.float32,
        )
        self.action_log_probs = np.zeros(
            (self.episode_limit, self.num_envs, action_shape),
            dtype=np.float32,
        )
        self.masks = np.ones(
            (self.episode_limit + 1, self.num_envs, num_agents, 1),
            dtype=np.float32,
        )
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0
