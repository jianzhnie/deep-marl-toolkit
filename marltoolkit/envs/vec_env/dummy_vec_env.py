from typing import Callable, List, Optional, Union

import gymnasium as gym
import numpy as np

from .base_vec_env import BaseVecEnv


class DummyVecEnv(BaseVecEnv):
    """Creates a simple vectorized wrapper for multiple environments, calling
    each environment in sequence on the current Python process. This is useful
    for computationally simple environment such as ``Cartpole-v1``, as the
    overhead of multiprocess or multithread outweighs the environment
    computation time. This can also be used for RL methods that require a
    vectorized environment, but that you want a single environments to train
    with.

    :param env_fns: a list of functions
        that return environments to vectorize
    :raises ValueError: If the same environment instance is passed as the output of two or more different env_fn.
    """

    actions: np.ndarray

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        super().__init__(
            len(env_fns),
            env.observation_space,
            env.share_observation_space,
            env.action_space,
        )
        self.actions = None

    def step_async(self, actions: Union[np.ndarray, List]) -> None:
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))
        self.actions = None
        return obs, rews, dones, infos

    def reset(self):
        obs = [env.reset() for env in self.envs]
        return np.array(obs)

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        """Gym environment rendering. If there are multiple environments then
        they are tiled together in one image via ``BaseVecEnv.render()``.

        :param mode: The rendering type.
        """
        return super().render(mode=mode)


class ShareDummyVecEnv(BaseVecEnv):

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        super().__init__(
            len(env_fns),
            env.observation_space,
            env.share_observation_space,
            env.action_space,
        )
        self.actions = None
        self.num_agents = env.num_agents

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, share_obs, rews, dones, infos, available_actions = map(
            np.array, zip(*results))
        return obs, share_obs, rews, dones, infos, available_actions

    def reset(self):
        results = [env.reset() for env in self.envs]
        obs, share_obs, available_actions = map(np.array, zip(*results))
        return obs, share_obs, available_actions

    def close(self):
        for env in self.envs:
            env.close()


class ChooseDummyVecEnv(BaseVecEnv):

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        super().__init__(
            len(env_fns),
            env.observation_space,
            env.share_observation_space,
            env.action_space,
        )
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, share_obs, rews, dones, infos, available_actions = map(
            np.array, zip(*results))
        self.actions = None
        return obs, share_obs, rews, dones, infos, available_actions

    def reset(self, reset_choose):
        results = [
            env.reset(choose) for (env, choose) in zip(self.envs, reset_choose)
        ]
        obs, share_obs, available_actions = map(np.array, zip(*results))
        return obs, share_obs, available_actions

    def close(self):
        for env in self.envs:
            env.close()
