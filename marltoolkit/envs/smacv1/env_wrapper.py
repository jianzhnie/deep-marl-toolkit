from copy import deepcopy

import numpy as np

from marltoolkit.envs.base_env import MultiAgentEnv
from marltoolkit.utils.transforms import OneHotTransform


class SC2EnvWrapper(object):

    def __init__(self, env: MultiAgentEnv):
        self.env = env
        env_info = env.get_env_info()
        self.episode_limit = env_info['episode_limit']
        self.n_actions = env_info['n_actions']
        self.num_agents = env_info['n_agents']
        self.state_shape = env_info['state_shape']
        self.obs_shape = env_info[
            'obs_shape'] + self.num_agents + self.n_actions
        self.agent_id_one_hot_transform = OneHotTransform(self.num_agents)
        self.actions_one_hot_transform = OneHotTransform(self.n_actions)
        self._init_agents_id_one_hot(self.num_agents)

    @property
    def win_counted(self):
        return self.env.win_counted

    def _init_agents_id_one_hot(self, n_agents):
        agents_id_one_hot = []
        for agent_id in range(n_agents):
            one_hot = self.agent_id_one_hot_transform(agent_id)
            agents_id_one_hot.append(one_hot)
        self.agents_id_one_hot = np.array(agents_id_one_hot)

    def _get_agents_id_one_hot(self):
        return deepcopy(self.agents_id_one_hot)

    def _get_actions_one_hot(self, actions):
        # (n_agents，n_actions)
        actions_one_hot = []
        for action in actions:
            one_hot = self.actions_one_hot_transform(action)
            actions_one_hot.append(one_hot)
        return np.array(actions_one_hot)

    def get_available_actions(self):
        # (n_agents，n_actions)
        available_actions = []
        for agent_id in range(self.num_agents):
            available_actions.append(
                self.env.get_avail_agent_actions(agent_id))
        return np.array(available_actions)

    def reset(self):
        self.env.reset()
        # action at last timestep
        # last_actions_one_hot shape: (self.n_agents, self.n_actions)
        last_actions_one_hot = np.zeros((self.num_agents, self.n_actions),
                                        dtype='float32')

        # obs shape: (self.n_agents, obs_dim)
        obs = np.array(self.env.get_obs())
        # agents_id_one_hot shape: (self.n_agents, self.n_agents)
        agents_id_one_hot = self._get_agents_id_one_hot()
        obs = np.concatenate([obs, last_actions_one_hot, agents_id_one_hot],
                             axis=-1)
        # obs shape: (self.n_agents, (obs_dim + self.n_actions + self.n_agents ))
        state = np.array(self.env.get_state())
        return (obs, state)

    def step(self, actions):
        reward, terminated, info = self.env.step(actions)

        state = np.array(self.env.get_state())
        last_actions_one_hot = self._get_actions_one_hot(actions)
        obs = np.array(self.env.get_obs())
        # obs shape: (self.n_agents, (obs_dim + self.n_actions + self.n_agents ))
        obs = np.concatenate(
            [obs, last_actions_one_hot, self.agents_id_one_hot], axis=-1)
        return obs, state, reward, terminated, info
