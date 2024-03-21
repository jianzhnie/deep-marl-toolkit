import time
import warnings
from typing import Any, Dict, List, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType

from .base_vec_env import BaseVecEnv, VecEnvWrapper


class Monitor(gym.Wrapper):
    """A monitor wrapper for Gym environments, it is used to know the episode
    reward, length, time and other data.

    :param env: The environment
    :param allow_early_resets: allows the reset of the environment before it is done
    :param reset_keywords: extra keywords for the reset call,
        if extra parameters are needed at reset
    :param info_keywords: extra information to log, from the information return of env.step()
    :param override_existing: appends to file if ``filename`` exists, otherwise
        override existing files (default)
    """

    def __init__(
            self,
            env: gym.Env,
            allow_early_resets: bool = True,
            reset_keywords: Tuple[str, ...] = (),
            info_keywords: Tuple[str, ...] = (),
    ):
        super().__init__(env=env)
        self.t_start = time.time()
        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards: List[float] = []
        self.needs_reset = True
        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_times: List[float] = []
        self.total_steps = 0
        # extra info about the current episode, that was passed in during reset()
        self.current_reset_info: Dict[str, Any] = {}

    def reset(self, **kwargs) -> Tuple[ObsType, Dict[str, Any]]:
        """Calls the Gym environment reset. Can only be called if the
        environment is over, or if allow_early_resets is True.

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError(
                'Tried to reset an environment before done. If you want to allow early resets, '
                'wrap your env with Monitor(env, path, allow_early_resets=True)'
            )
        self.rewards = []
        self.needs_reset = False
        for key in self.reset_keywords:
            value = kwargs.get(key)
            if value is None:
                raise ValueError(
                    f'Expected you to pass keyword argument {key} into reset')
            self.current_reset_info[key] = value
        return self.env.reset(**kwargs)

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Step the environment with the given action.

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        if self.needs_reset:
            raise RuntimeError('Tried to step environment that needs reset')
        observation, reward, terminated, truncated, info = self.env.step(
            action)
        self.rewards.append(float(reward))
        if terminated or truncated:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {
                'r': round(ep_rew, 6),
                'l': ep_len,
                't': round(time.time() - self.t_start, 6),
            }
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            info['episode'] = ep_info
        self.total_steps += 1
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        """Closes the environment."""
        super().close()

    def get_total_steps(self) -> int:
        """Returns the total number of timesteps.

        :return:
        """
        return self.total_steps

    def get_episode_rewards(self) -> List[float]:
        """Returns the rewards of all the episodes.

        :return:
        """
        return self.episode_returns

    def get_episode_lengths(self) -> List[int]:
        """Returns the number of timesteps of all the episodes.

        :return:
        """
        return self.episode_lengths

    def get_episode_times(self) -> List[float]:
        """Returns the runtime in seconds of all the episodes.

        :return:
        """
        return self.episode_times


class VecMonitor(VecEnvWrapper):
    """A vectorized monitor wrapper for *vectorized* Gym environments, it is
    used to record the episode reward, length, time and other data.

    Some environments like `openai/procgen <https://github.com/openai/procgen>`_
    or `gym3 <https://github.com/openai/gym3>`_ directly initialize the
    vectorized environments, without giving us a chance to use the ``Monitor``
    wrapper. So this class simply does the job of the ``Monitor`` wrapper on
    a vectorized level.

    :param vec_env: The vectorized environment
    :param filename: the location to save a log file, can be None for no log
    :param info_keywords: extra information to log, from the information return of env.step()
    """

    def __init__(
            self,
            vec_env: BaseVecEnv,
            info_keywords: Tuple[str, ...] = (),
    ) -> None:
        # This check is not valid for special `VecEnv`
        # like the ones created by Procgen, that does follow completely
        # the `VecEnv` interface
        try:
            is_wrapped_with_monitor = vec_env.env_is_wrapped(Monitor)[0]
        except AttributeError:
            is_wrapped_with_monitor = False

        if is_wrapped_with_monitor:
            warnings.warn(
                'The environment is already wrapped with a `Monitor` wrapper'
                'but you are wrapping it with a `VecMonitor` wrapper, the `Monitor` statistics will be'
                'overwritten by the `VecMonitor` ones.',
                UserWarning,
            )

        VecEnvWrapper.__init__(self, vec_env)
        self.episode_count = 0
        self.t_start = time.time()

        self.info_keywords = info_keywords
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

    def reset(self):
        obs = self.vec_env.reset()
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return obs

    def step_wait(self):
        obs, rewards, dones, infos = self.vec_env.step_wait()
        self.episode_returns += rewards
        self.episode_lengths += 1
        new_infos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_info = {
                    'r': episode_return,
                    'l': episode_length,
                    't': round(time.time() - self.t_start, 6),
                }
                for key in self.info_keywords:
                    episode_info[key] = info[key]
                info['episode'] = episode_info
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
                new_infos[i] = info
        return obs, rewards, dones, new_infos

    def close(self) -> None:
        return self.vec_env.close()
