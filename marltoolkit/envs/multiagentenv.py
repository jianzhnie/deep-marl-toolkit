from typing import Any, Dict, Optional, Set, Tuple

import gymnasium as gym


class MARLBaseEnv(gym.Env):
    """An environment that hosts multiple independent agents.

    Agents are identified by (string) agent ids. Note that these "agents" here
    are not to be confused with RLlib Algorithms, which are also sometimes
    referred to as "agents" or "RL agents".

    The preferred format for action- and observation space is a mapping from agent
    ids to their individual spaces. If that is not provided, the respective methods'
    observation_space_contains(), action_space_contains(),
    action_space_sample() and observation_space_sample() have to be overwritten.
    """

    def __init__(self):
        if not hasattr(self, 'observation_space'):
            self.observation_space = None
        if not hasattr(self, 'action_space'):
            self.action_space = None
        if not hasattr(self, '_agent_ids'):
            self._agent_ids = set()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
        """Resets the env and returns observations from ready agents.

        Args:
            seed: An optional seed to use for the new episode.

        Returns:
            New observations for each ready agent.

        .. testcode::
            :skipif: True

            from ray.rllib.env.multi_agent_env import MultiAgentEnv
            class MyMultiAgentEnv(MultiAgentEnv):
                # Define your env here.
            env = MyMultiAgentEnv()
            obs, infos = env.reset(seed=42, options={})
            print(obs)

        .. testoutput::

            {
                "car_0": [2.4, 1.6],
                "car_1": [3.4, -3.2],
                "traffic_light_1": [0, 3, 5, 1],
            }
        """
        # Call super's `reset()` method to (maybe) set the given `seed`.
        super().reset(seed=seed, options=options)

    def step(
        self, action_dict: Dict[Any, Any]
    ) -> Tuple[Dict[Any, Any], Dict[Any, Any], Dict[Any, Any], Dict[Any, Any],
               Dict[Any, Any]]:
        """Returns observations from ready agents.

        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.

        Returns:
            Tuple containing 1) new observations for
            each ready agent, 2) reward values for each ready agent. If
            the episode is just started, the value will be None.
            3) Terminated values for each ready agent. The special key
            "__all__" (required) is used to indicate env termination.
            4) Truncated values for each ready agent.
            5) Info values for each agent id (may be empty dicts).

        .. testcode::
            :skipif: True

            env = ...
            obs, rewards, terminateds, truncateds, infos = env.step(action_dict={
                "car_0": 1, "car_1": 0, "traffic_light_1": 2,
            })
            print(rewards)

            print(terminateds)

            print(infos)

        .. testoutput::

            {
                "car_0": 3,
                "car_1": -1,
                "traffic_light_1": 0,
            }
            {
                "car_0": False,    # car_0 is still running
                "car_1": True,     # car_1 is terminated
                "__all__": False,  # the env is not terminated
            }
            {
                "car_0": {},  # info for car_0
                "car_1": {},  # info for car_1
            }
        """
        raise NotImplementedError

    def get_agent_ids(self) -> Set[Any]:
        """Returns a set of agent ids in the environment.

        Returns:
            Set of agent ids.
        """
        if not isinstance(self._agent_ids, set):
            self._agent_ids = set(self._agent_ids)
        return self._agent_ids


class MultiAgentEnv(object):
    """A multi-agent environment wrapper."""

    def __init__(self):
        self.n_agents = None
        self.episode_limit = None

    def step(self, actions):
        """Returns reward, terminated, info."""
        raise NotImplementedError()

    def get_obs(self):
        """Returns all agent observations in a list."""
        raise NotImplementedError()

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        raise NotImplementedError()

    def get_obs_size(self):
        """Returns the shape of the observation."""
        raise NotImplementedError()

    def get_state(self):
        raise NotImplementedError()

    def get_state_size(self):
        """Returns the shape of the state."""
        raise NotImplementedError()

    def get_avail_actions(self):
        raise NotImplementedError()

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        raise NotImplementedError()

    def get_available_actions(self):
        raise NotImplementedError()

    def _get_actions_one_hot(self):
        raise NotImplementedError()

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        raise NotImplementedError()

    def reset(self):
        """Returns initial observations and states."""
        raise NotImplementedError()

    def render(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def seed(self):
        raise NotImplementedError()

    def save_replay(self):
        raise NotImplementedError()

    def get_env_info(self):
        env_info = {
            'state_shape': self.get_state_size(),
            'obs_shape': self.get_obs_size(),
            'n_actions': self.get_total_actions(),
            'n_agents': self.n_agents,
            'episode_limit': self.episode_limit
        }
        return env_info
