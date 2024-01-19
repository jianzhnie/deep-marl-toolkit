'''
Author: jianzhnie
LastEditors: jianzhnie
Description: RLToolKit is a flexible and high-efficient reinforcement learning framework.
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''
from copy import deepcopy
from typing import List, Tuple, Union

import numpy as np
from gymnasium.spaces import Box, Dict, Discrete
from smac.env import StarCraft2Env

from marltoolkit.envs.multiagentenv import MARLBaseEnv
from marltoolkit.utils.transforms import OneHotTransform


class MARLSMACEnv(object):
    """Wrapper for StarCraft2Env providing a more user-friendly interface."""

    def __init__(self, map_name: str = '3m'):
        """Initialize the SC2Env.

        Parameters:
        - map_name (str): Name of the map for the StarCraft2 environment.
        """
        self.env = StarCraft2Env(map_name=map_name)
        self.env_info = self.env.get_env_info()

        # Number of agents and enemies
        self.num_agents = self.env_info['num_agents']
        self.num_enemies = self.env.n_enemies
        self.agents = ['agent_{}'.format(i) for i in range(self.num_agents)]

        # Number of actions
        self.action_shape = self.env_info['n_actions']
        self.n_actions = self.env_info['n_actions']

        # State and observation shapes
        self.state_shape = self.env_info['state_shape']
        self.obs_shape = self.env_info['obs_shape']

        # Reward and done shapes
        self.dim_reward = self.num_agents
        self.dim_done = self.num_agents

        # Space
        self.state_space = Box(-2.0, 2.0, shape=(self.state_shape, ))
        self.obs_space = Box(-2.0, 2.0, shape=(self.obs_shape, ))
        self.action_mask_space = Box(-2.0, 2.0, shape=(self.n_actions, ))
        self.action_space = Discrete(self.n_actions)

        self.reward_space = (self.dim_reward, )
        self.done_space = (self.dim_done, )

        # Max episode steps
        self.episode_limit = self.env_info['episode_limit']

        # Observation concatenation shape
        self.obs_concate_shape = self.obs_shape + self.num_agents + self.n_actions

        # One-hot transformations
        self.agent_id_one_hot_transform = OneHotTransform(self.num_agents)
        self.actions_one_hot_transform = OneHotTransform(self.n_actions)

        # Initialize agent IDs one-hot encoding
        self._init_agents_id_one_hot(self.num_agents)

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

    def _init_agents_id_one_hot(self, num_agents: int):
        """Initialize the one-hot encoding for agent IDs.

        Parameters:
        - num_agents (int): Number of agents.
        """
        agents_id_one_hot = [
            self.agent_id_one_hot_transform(agent_id)
            for agent_id in range(num_agents)
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
        self.env.reset()
        obs_smac = self.env.get_obs()
        state_smac = self.env.get_state()
        last_actions_one_hot = np.zeros((self.num_agents, self.n_actions),
                                        dtype='float32')
        agents_id_one_hot = self._get_agents_id_one_hot()
        obs_concate = np.concatenate(
            [obs_smac, last_actions_one_hot, agents_id_one_hot], axis=-1)

        self._episode_step = 0
        self._episode_score = 0.0
        info = {
            'episode_step': self._episode_step,
            'episode_score': self._episode_score,
        }
        return state_smac, obs_smac, obs_concate, info

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

        obs_smac = self.env.get_obs()
        state_smac = self.env.get_state()
        reward_n = np.array([[reward] for _ in range(self.num_agents)])

        last_actions_one_hot = self._get_actions_one_hot(actions)

        self._episode_step += 1
        self._episode_score += reward

        info['episode_step'] = self._episode_step
        info['episode_score'] = self._episode_score

        truncated = True if self._episode_step >= self.episode_limit else False
        obs_concate = np.concatenate(
            [obs_smac, last_actions_one_hot, self.agents_id_one_hot], axis=-1)

        return state_smac, obs_smac, obs_concate, reward_n, [terminated
                                                             ], [truncated
                                                                 ], info

    def get_env_info(self):
        """Get the environment information."""
        env_info = {
            'state_space': self.state_space,
            'obs_space': self.obs_space,
            'action_mask': self.action_mask_space,
            'action_space': self.action_space,
            'num_agents': self.num_agents,
            'episode_limit': self.episode_limit,
        }
        return env_info


class RLlibSMAC(MARLBaseEnv):

    def __init__(self, map_name):
        map_name = map_name if isinstance(map_name,
                                          str) else map_name['map_name']
        self.env = StarCraft2Env(map_name)

        env_info = self.env.get_env_info()
        self.num_agents = self.env.n_agents
        self.agents = ['agent_{}'.format(i) for i in range(self.num_agents)]
        obs_shape = env_info['obs_shape']
        n_actions = env_info['n_actions']
        state_shape = env_info['state_shape']
        self.observation_space = Dict({
            'obs':
            Box(-2.0, 2.0, shape=(obs_shape, )),
            'state':
            Box(-2.0, 2.0, shape=(state_shape, )),
            'action_mask':
            Box(-2.0, 2.0, shape=(n_actions, ))
        })
        self.action_space = Discrete(n_actions)

    def reset(self):
        self.env.reset()
        obs_smac = self.env.get_obs()
        state_smac = self.env.get_state()
        obs_dict = {}
        for agent_index in range(self.num_agents):
            obs_one_agent = obs_smac[agent_index]
            state_one_agent = state_smac
            action_mask_one_agent = np.array(
                self.env.get_avail_agent_actions(agent_index)).astype(
                    np.float32)
            agent_index = 'agent_{}'.format(agent_index)
            obs_dict[agent_index] = {
                'obs': obs_one_agent,
                'state': state_one_agent,
                'action_mask': action_mask_one_agent,
            }

        return obs_dict

    def step(self, actions):

        actions_ls = [int(actions[agent_id]) for agent_id in actions.keys()]

        reward, terminated, info = self.env.step(actions_ls)

        obs_smac = self.env.get_obs()
        state_smac = self.env.get_state()

        obs_dict = {}
        reward_dict = {}
        for agent_index in range(self.num_agents):
            obs_one_agent = obs_smac[agent_index]
            state_one_agent = state_smac
            action_mask_one_agent = np.array(
                self.env.get_avail_agent_actions(agent_index)).astype(
                    np.float32)
            agent_index = 'agent_{}'.format(agent_index)
            obs_dict[agent_index] = {
                'obs': obs_one_agent,
                'state': state_one_agent,
                'action_mask': action_mask_one_agent
            }
            reward_dict[agent_index] = reward

        dones = {'__all__': terminated}

        return obs_dict, reward_dict, dones, {}

    def get_env_info(self):
        env_info = {
            'space_obs': self.observation_space,
            'space_act': self.action_space,
            'num_agents': self.num_agents,
            'episode_limit': self.env.episode_limit,
        }
        return env_info

    def close(self):
        self.env.close()
