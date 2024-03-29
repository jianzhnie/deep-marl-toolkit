from typing import Optional

import gymnasium as gym


class MultiAgentEnv(gym.Env):
    """A multi-agent environment wrapper. An environment that hosts multiple
    independent agents.

    Agents are identified by (string) agent ids. Note that these "agents" here
    are not to be confused with RLlib Algorithms, which are also sometimes
    referred to as "agents" or "RL agents".

    The preferred format for action- and observation space is a mapping from agent
    ids to their individual spaces. If that is not provided, the respective methods'
    observation_space_contains(), action_space_contains(),
    action_space_sample() and observation_space_sample() have to be overwritten.
    """

    def __init__(self) -> None:
        self.num_agents = None
        self.episode_limit = None
        if not hasattr(self, 'obs_space'):
            self.observation_space = None
        if not hasattr(self, 'state_space'):
            self.observation_space = None
        if not hasattr(self, 'action_space'):
            self.action_space = None
        if not hasattr(self, '_agent_ids'):
            self._agent_ids = set()

    def step(self, actions):
        """Returns reward, terminated, info."""
        raise NotImplementedError

    def get_obs(self):
        """Returns all agent observations in a list."""
        raise NotImplementedError

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        raise NotImplementedError

    def get_obs_size(self):
        """Returns the shape of the observation."""
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def get_state_size(self):
        """Returns the shape of the state."""
        raise NotImplementedError

    def get_avail_actions(self):
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        raise NotImplementedError

    def get_available_actions(self):
        raise NotImplementedError

    def _get_actions_one_hot(self):
        raise NotImplementedError

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        raise NotImplementedError

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """Returns initial observations and states."""
        super().reset(seed=seed, options=options)

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self):
        raise NotImplementedError

    def save_replay(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {
            'obs_shape': self.get_obs_size(),
            'state_shape': self.get_state_size(),
            'n_actions': self.get_total_actions(),
            'num_agents': self.num_agents,
            'episode_limit': self.episode_limit,
        }
        return env_info
