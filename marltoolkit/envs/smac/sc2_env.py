'''
Author: jianzhnie
LastEditors: jianzhnie
Description: RLToolKit is a flexible and high-efficient reinforcement learning framework.
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''
from copy import deepcopy
from typing import List, Tuple, Union

import numpy as np
from smac.env import StarCraft2Env

from marltoolkit.utils.transforms import OneHotTransform


class SC2Env:
    """Wrapper for StarCraft2Env providing a more user-friendly interface."""

    def __init__(self, map_name: str = '3m'):
        """Initialize the SC2Env.

        Parameters:
        - map_name (str): Name of the map for the StarCraft2 environment.
        """
        self.env = StarCraft2Env(map_name=map_name)
        self.env_info = self.env.get_env_info()

        # Number of agents and enemies
        self.n_agents = self.env_info['n_agents']
        self.n_enemies = self.env.n_enemies

        # Number of actions
        self.action_shape = self.env_info['n_actions']
        self.n_actions = self.env_info['n_actions']

        # State and observation shapes
        self.state_shape = self.env_info['state_shape']
        self.obs_shape = self.env_info['obs_shape']

        # Reward and done shapes
        self.dim_reward = self.n_agents
        self.dim_done = self.n_agents

        # Space
        self.obs_space = (self.obs_shape, )
        self.act_space = (self.n_actions, )
        self.reward_space = (self.dim_reward, )
        self.done_space = (self.dim_done, )

        # Max episode steps
        self.episode_limit = self.env_info['episode_limit']

        # Observation concatenation shape
        self.obs_concate_shape = self.obs_shape + self.n_agents + self.n_actions

        # One-hot transformations
        self.agent_id_one_hot_transform = OneHotTransform(self.n_agents)
        self.actions_one_hot_transform = OneHotTransform(self.n_actions)

        # Initialize agent IDs one-hot encoding
        self._init_agents_id_one_hot(self.n_agents)

        # Buffer information
        self.buf_info = {
            'battle_won': 0,
            'dead_allies': 0,
            'dead_enemies': 0,
        }

        # Episode variables
        self._episode_step = 0
        self._episode_score = 0

    @property
    def win_counted(self):
        """Check if the win is counted."""
        return self.env.win_counted

    def _init_agents_id_one_hot(self, n_agents: int):
        """Initialize the one-hot encoding for agent IDs.

        Parameters:
        - n_agents (int): Number of agents.
        """
        agents_id_one_hot = [
            self.agent_id_one_hot_transform(agent_id)
            for agent_id in range(n_agents)
        ]
        self.agents_id_one_hot = np.array(agents_id_one_hot)

    def _get_agents_id_one_hot(self) -> np.ndarray:
        """Get the one-hot encoding for agent IDs."""
        return deepcopy(self.agents_id_one_hot)

    def _get_actions_one_hot(
            self, actions: Union[np.ndarray, List[int]]) -> np.ndarray:
        """Get the one-hot encoding for a list of actions.

        Parameters:
        - actions (Union[np.ndarray, List[int]]): List of actions.

        Returns:
        - np.ndarray: One-hot encoding of actions.
        """
        actions_one_hot = [
            self.actions_one_hot_transform(action) for action in actions
        ]
        return np.array(actions_one_hot)

    def get_available_actions(self):
        return self.env.get_avail_actions()

    def close(self):
        """Close the environment."""
        self.env.close()

    def render(self, mode):
        """Render the environment.

        Parameters:
        - mode: Rendering mode.
        """
        return self.env.render(mode)

    def reset(self) -> Tuple:
        """Reset the environment.

        Returns:
        - Tuple: Tuple containing state, observation, concatenated observation, and info.
        """
        obs, state = self.env.reset()
        state = np.array(state)
        obs = np.array(obs)
        last_actions_one_hot = np.zeros((self.n_agents, self.n_actions),
                                        dtype='float32')
        agents_id_one_hot = self._get_agents_id_one_hot()
        obs_concate = np.concatenate(
            [obs, last_actions_one_hot, agents_id_one_hot], axis=-1)

        self._episode_step = 0
        self._episode_score = 0.0
        info = {
            'episode_step': self._episode_step,
            'episode_score': self._episode_score,
        }

        return state, obs, obs_concate, info

    def step(self, actions: Union[np.ndarray, List[int]]) -> Tuple:
        """Take a step in the environment.

        Parameters:
        - actions (Union[np.ndarray, List[int]]): List of actions.

        Returns:
        - Tuple: Tuple containing state, observation, concatenated observation, reward, termination flag, truncation flag, and info.
        """
        reward, terminated, info = self.env.step(actions)

        if not info:
            info = self.buf_info

        obs = np.array(self.env.get_obs())
        state = np.array(self.env.get_state())
        reward_n = np.array([[reward] for _ in range(self.n_agents)])

        last_actions_one_hot = self._get_actions_one_hot(actions)

        self._episode_step += 1
        self._episode_score += reward
        self.buf_info = deepcopy(info)

        info['episode_step'] = self._episode_step
        info['episode_score'] = self._episode_score

        truncated = True if self._episode_step >= self.episode_limit else False
        obs_concate = np.concatenate(
            [obs, last_actions_one_hot, self.agents_id_one_hot], axis=-1)

        return state, obs, obs_concate, reward_n, [terminated], [truncated
                                                                 ], info
