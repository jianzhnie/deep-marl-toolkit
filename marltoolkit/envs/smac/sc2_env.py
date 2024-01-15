'''
Author: jianzhnie
LastEditors: jianzhnie
Description: RLToolKit is a flexible and high-efficient reinforcement learning framework.
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''
import copy
from ast import Tuple
from copy import deepcopy
from typing import List

import numpy as np
from smac.env import StarCraft2Env

from marltoolkit.utils.transforms import OneHotTransform


class SC2Env(object):

    def __init__(self, map_name: str = '3m'):
        self.env = StarCraft2Env(map_name=map_name)
        env_info = self.env.get_env_info()

        self.n_agents = env_info['n_agents']
        self.action_shape = env_info['n_actions']
        self.n_actions = env_info['n_actions']
        self.state_shape = env_info['state_shape']
        self.obs_shape = env_info['obs_shape']
        self.dim_reward = self.n_agents

        self.obs_space = (self.obs_shape, )
        self.act_space = (self.n_actions, )
        self.episode_limit = env_info['episode_limit']

        self.obs_shape_center = env_info[
            'obs_shape'] + self.n_agents + self.n_actions
        self.agent_id_one_hot_transform = OneHotTransform(self.n_agents)
        self.actions_one_hot_transform = OneHotTransform(self.n_actions)
        self._init_agents_id_one_hot(self.n_agents)
        self.buf_info = {
            'battle_won': 0,
            'dead_allies': 0,
            'dead_enemies': 0,
        }
        self._episode_step = 0
        self._episode_score = 0

    @property
    def win_counted(self):
        return self.env.win_counted

    def _init_agents_id_one_hot(self, n_agents: int):
        agents_id_one_hot = []
        for agent_id in range(n_agents):
            one_hot = self.agent_id_one_hot_transform(agent_id)
            agents_id_one_hot.append(one_hot)
        self.agents_id_one_hot = np.array(agents_id_one_hot)

    def _get_agents_id_one_hot(self) -> np.ndarray:
        return deepcopy(self.agents_id_one_hot)

    def _get_actions_one_hot(self, actions) -> np.ndarray:
        # (n_agents，n_actions)
        actions_one_hot = []
        for action in actions:
            one_hot = self.actions_one_hot_transform(action)
            actions_one_hot.append(one_hot)
        return np.array(actions_one_hot)

    def get_available_actions(self) -> np.ndarray:
        # (n_agents，n_actions)
        available_actions = []
        for agent_id in range(self.n_agents):
            avail_agent_action = self.env.get_avail_agent_actions(agent_id)
            available_actions.append(avail_agent_action)
        return np.array(available_actions)

    def close(self):
        self.env.close()

    def render(self, mode):
        return self.env.render(mode)

    def reset(self) -> Tuple:
        obs, state = self.env.reset()
        # state shape: (self.n_agents, self.state_shape)
        state = np.array(state)
        # obs shape: (self.n_agents, obs_dim)
        obs = np.array(obs)
        # action at last timestep
        # last_actions_one_hot shape: (self.n_agents, self.n_actions)
        last_actions_one_hot = np.zeros((self.n_agents, self.n_actions),
                                        dtype='float32')
        # agents_id_one_hot shape: (self.n_agents, self.n_agents)
        agents_id_one_hot = self._get_agents_id_one_hot()
        # obs shape: (self.n_agents, (obs_dim + self.n_actions + self.n_agents ))
        obs_concate = np.concatenate(
            [obs, last_actions_one_hot, agents_id_one_hot], axis=-1)
        self._episode_step = 0
        self._episode_score = 0.0
        info = {
            'episode_step': self._episode_step,
            'episode_score': self._episode_score,
        }
        return state, obs_concate, info

    def step(self, actions: List[int]) -> Tuple:
        reward, terminated, info = self.env.step(actions)
        if info == {}:
            info = self.buf_info

        obs = np.array(self.env.get_obs())
        state = np.array(self.env.get_state())
        reward_n = np.array([[reward] for _ in range(self.n_agents)])
        last_actions_one_hot = self._get_actions_one_hot(actions)

        self._episode_step += 1
        self._episode_score += reward
        self.buf_info = copy.deepcopy(info)
        info['episode_step'] = self._episode_step
        info['episode_score'] = self._episode_score

        truncated = True if self._episode_step >= self.episode_limit else False
        # obs shape: (self.n_agents, (obs_dim + self.n_actions + self.n_agents ))
        concate_obs = np.concatenate(
            [obs, last_actions_one_hot, self.agents_id_one_hot], axis=-1)
        return state, obs, concate_obs, reward_n, [terminated], [truncated
                                                                 ], info
