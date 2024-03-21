from typing import List, Tuple, Union

import numpy as np
from gymnasium.spaces import Box, Dict, Discrete
from smac.env import StarCraft2Env

from marltoolkit.envs.marl_base_env import MARLBaseEnv


class SMACWrapperEnv(object):
    """Wrapper for StarCraft2Env providing a more user-friendly interface."""

    def __init__(self, map_name: str, **kwargs):
        """Initialize the SC2Env.

        Parameters:
        - map_name (str): Name of the map for the StarCraft2 environment.
        """
        self.env = StarCraft2Env(map_name=map_name, **kwargs)
        self.env_info = self.env.get_env_info()

        # Number of agents and enemies
        self.num_agents = self.env.n_agents
        self.num_enemies = self.env.n_enemies

        # Number of actions
        self.action_shape = self.env_info['n_actions']
        self.n_actions = self.env_info['n_actions']

        # State and observation shapes
        self.obs_shape = self.env_info['obs_shape']
        self.state_shape = self.env_info['state_shape']

        # Reward and done shapes
        self.reward_dim = self.num_agents
        self.done_dim = self.num_agents

        # Space
        self.obs_space = Box(-2.0, 2.0, shape=(self.obs_shape, ))
        self.state_space = Box(-2.0, 2.0, shape=(self.state_shape, ))
        self.action_mask_space = Box(-2.0, 2.0, shape=(self.n_actions, ))
        self.action_space = Discrete(self.n_actions)

        self.reward_space = (self.reward_dim, )
        self.done_space = (self.done_dim, )

        # Max episode steps
        self.episode_limit = self.env_info['episode_limit']
        self.filled = np.zeros([self.episode_limit, 1], bool)

        # Buffer information
        self.buf_info = {
            'battle_won': 0,
            'dead_allies': 0,
            'dead_enemies': 0,
        }

        # Episode variables
        self._episode_step = 0
        self._episode_score = 0

        # Render Mode
        self.render_mode = 'human'

    @property
    def win_counted(self):
        """Check if the win is counted."""
        return self.env.win_counted

    def get_available_actions(self):
        return self.env.get_avail_actions()

    def close(self):
        """Close the environment."""
        self.env.close()

    def render(self, mode: str):
        """Render the environment.

        Parameters:
        - mode: Rendering mode.
        """
        return self.env.render(mode)

    def reset(self, seed: int = None):
        """Reset the environment.

        Returns:
        - Tuple: Tuple containing state, observation, concatenated observation, and info.
        """
        obs, state = self.env.reset()
        self._episode_step = 0
        self._episode_score = 0.0
        info = {
            'episode_step': self._episode_step,
            'episode_score': self._episode_score,
        }
        return obs, state, info

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

        obs = self.env.get_obs()
        state = self.env.get_state()
        reward_n = np.array([[reward] for _ in range(self.num_agents)])

        self._episode_step += 1
        self._episode_score += reward

        info['episode_step'] = self._episode_step
        info['episode_score'] = self._episode_score

        truncated = True if self._episode_step >= self.episode_limit else False
        return obs, state, reward_n, terminated, truncated, info

    def get_env_info(self):
        """Get the environment information."""
        env_info = {
            'obs_shape': self.obs_shape,
            'obs_space': self.obs_space,
            'state_shape': self.state_shape,
            'state_space': self.state_space,
            'action_mask_space': self.action_mask_space,
            'n_actions': self.n_actions,
            'action_space': self.action_space,
            'num_agents': self.num_agents,
            'episode_limit': self.episode_limit,
        }
        return env_info


class RLlibSMAC(MARLBaseEnv):

    def __init__(self, map_name):
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
