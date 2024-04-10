from typing import List, Tuple, TypeVar, Union

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete
from smac.env import StarCraft2Env

from marltoolkit.envs.base_env import MultiAgentEnv
from marltoolkit.utils.transforms import OneHotTransform

AgentID = TypeVar('AgentID')


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
        self.agents = ['agent_{}'.format(i) for i in range(self.num_agents)]

        # Number of actions
        self.action_dim = self.env_info['n_actions']
        self.n_actions = self.env_info['n_actions']

        # State and observation dims
        self.obs_dim = self.env_info['obs_shape']
        self.state_dim = self.env_info['state_shape']

        # Reward and done shapes
        self.reward_dim = 1
        self.done_dim = 1

        # Observation and state shapes
        self.obs_shape = (self.obs_dim, )
        self.action_shape = (self.action_dim, )
        self.state_shape = (self.state_dim, )
        self.reward_shape = (self.reward_dim, )
        self.done_shape = (self.done_dim, )

        # Space
        self.obs_space = Box(
            -np.inf,
            np.inf,
            shape=(
                self.num_agents,
                self.obs_dim,
            ),
        )
        self.state_space = Box(-np.inf, np.inf, shape=(self.state_dim, ))
        self.action_space = Discrete(self.n_actions)

        # Multi-Agent Sapces
        self.obs_spaces: dict[AgentID, gym.spaces.Space] = {}
        self.action_spaces: dict[AgentID, gym.spaces.Space] = {}
        for agent_id in self.agents:
            self.obs_spaces[agent_id] = Box(
                -np.inf,
                np.inf,
                shape=(self.obs_dim, ),
            )
            self.action_spaces[agent_id] = Discrete(self.n_actions)

        # Max episode steps
        self.episode_limit = self.env_info['episode_limit']
        self.filled = np.zeros([self.episode_limit, 1], bool)

        # one-hot transform
        self.agent_id_one_hot_transform = OneHotTransform(self.num_agents)
        self.actions_one_hot_transform = OneHotTransform(self.n_actions)

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

    def seed(self, seed: int = None):
        """Set the seed for the environment."""
        self.env.seed()

    def get_available_actions(self):
        """Get the available actions.

        Returns:
            - np.ndarray: Array of available actions.
                shape: (num_agents, n_actions)
        """
        return self.env.get_avail_actions()

    def get_available_agent_actions(self):
        action_dict = {}
        for agent_index in range(self.num_agents):
            agent_index = 'agent_{}'.format(agent_index)
            action_dict[agent_index] = self.env.get_avail_agent_actions(
                agent_index)
        return action_dict

    def get_agents_id_one_hot(self):
        """Get the one-hot encoding of the agent IDs.

        Args:
            num_agents (int): Number of agents.

        Returns:
            np.ndarray: Shape : (num_agents, num_agents)
                One-hot encoding of the agent IDs.
        """
        agents_id_one_hot = []
        for agent_id in range(self.num_agents):
            one_hot = self.agent_id_one_hot_transform(agent_id)
            agents_id_one_hot.append(one_hot)
        return np.array(agents_id_one_hot)

    def get_actions_one_hot(self, actions: List[int]):
        """Get the one-hot encoding of the actions.

        Args:
            actions (List[int]): List of actions. len(actions) = num_agents

        Returns:
            np.ndarray: Shape : (num_agents, n_actions)
                One-hot encoding of the actions.
        """
        actions_one_hot = []
        for action in actions:
            one_hot = self.actions_one_hot_transform(action)
            actions_one_hot.append(one_hot)
        return np.array(actions_one_hot)

    def get_actor_input_shape(
        self,
        use_global_state: bool = False,
        use_last_action: bool = False,
        use_agent_id_onehot: bool = False,
    ) -> None:
        """Get the input shape of the actor model.

        Args:
            args (argparse.Namespace): The arguments
        Returns:
            input_shape (int): The input shape of the actor model.
        """
        input_shape = self.obs_dim
        if use_global_state:
            input_shape += self.state_shape
        if use_last_action:
            input_shape += self.n_actions
        if use_agent_id_onehot:
            input_shape += self.num_agents
        return input_shape

    def get_critic_input_shape(
        self,
        use_global_state: bool = False,
        use_last_action: bool = False,
        use_agent_id_onehot: bool = False,
    ) -> None:
        """Get the input shape of the critic model.

        Args:
            args (argparse.Namespace): The arguments.

        Returns:
            input_shape (int): The input shape of the critic model.
        """
        input_shape = self.obs_shape
        if use_global_state:
            input_shape += self.state_shape
        if use_last_action:
            input_shape += self.n_actions
        if use_agent_id_onehot:
            input_shape += self.num_agents
        return input_shape

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
        - Tuple: state, observation and info.
        """
        obs, state = self.env.reset()
        obs_dict, state_dict = {}, {}
        for agent_index in range(self.num_agents):
            obs_one_agent = obs[agent_index]
            state_one_agent = state
            agent_index = 'agent_{}'.format(agent_index)
            obs_dict[agent_index] = obs_one_agent
            state_dict[agent_index] = state_one_agent

        self._episode_step = 0
        self._episode_score = 0
        info = {
            'episode_step': self._episode_step,
            'episode_score': self._episode_score,
        }
        return np.array(obs), np.array(state), info

    def step(self, actions: Union[np.ndarray, List[int]]) -> Tuple:
        """Take a step in the environment.

        Parameters:
            - actions (Union[np.ndarray, List[int]]): List of actions.

        Returns:
            - Tuple: state, observation, reward, termination flag, truncation flag, and info.
        """

        reward, terminated, info = self.env.step(actions)

        if not info:
            info = self.buf_info

        obs = self.env.get_obs()
        state = self.env.get_state()

        self._episode_step += 1
        self._episode_score += reward

        info['episode_step'] = self._episode_step
        info['episode_score'] = self._episode_score

        truncated = True if self._episode_step >= self.episode_limit else False
        return (
            np.array(obs),
            np.array(state),
            np.array(reward),
            terminated,
            truncated,
            info,
        )

    def get_obs_space(self, agent: AgentID) -> gym.spaces.Space:
        """Takes in agent and returns the observation space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the observation_spaces dict
        """
        return self.obs_spaces[agent]

    def get_action_space(self, agent: AgentID) -> gym.spaces.Space:
        """Takes in agent and returns the action space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the action_spaces dict
        """
        return self.action_spaces[agent]

    def get_env_info(self):
        """Get the environment information."""
        env_info = {
            'obs_shape': self.obs_shape,
            'obs_space': self.obs_space,
            'state_shape': self.state_shape,
            'state_space': self.state_space,
            'n_actions': self.n_actions,
            'action_space': self.action_space,
            'num_agents': self.num_agents,
            'episode_limit': self.episode_limit,
        }
        return env_info


class RLlibSMAC(MultiAgentEnv):

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
            Box(-2.0, 2.0, shape=(n_actions, )),
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
                'action_mask': action_mask_one_agent,
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
