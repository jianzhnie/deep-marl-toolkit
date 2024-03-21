import warnings
from collections import OrderedDict
from copy import deepcopy
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Tuple, Type, Union)

import gymnasium as gym
import numpy as np

from .base_vec_env import BaseVecEnv
from .utils import copy_obs_dict, dict_to_obs, is_wrapped, obs_space_info


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
            env.obs_space,
            env.state_space,
            env.action_space,
        )
        self.actions = None
        self.obs_keys, obs_shapes, obs_dtypes = obs_space_info(env.obs_space)
        self.state_keys, state_shapes, state_dtypes = obs_space_info(
            env.state_space)

        self.buf_obs = OrderedDict([(
            k,
            np.zeros((self.num_envs, *tuple(obs_shapes[k])),
                     dtype=obs_dtypes[k]),
        ) for k in self.obs_keys])
        self.buf_state = OrderedDict([(
            k,
            np.zeros((self.num_envs, *tuple(state_shapes[k])),
                     dtype=state_dtypes[k]),
        ) for k in self.state_keys])
        self.buf_dones = np.zeros((self.num_envs, ), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs, ), dtype=np.float32)
        self.buf_infos: List[Dict[str,
                                  Any]] = [{} for _ in range(self.num_envs)]
        self.metadata = env.metadata

    def step_async(self, actions: Union[np.ndarray, List]) -> None:
        self.actions = actions

    def step_wait(self):
        for env_idx in range(self.num_envs):
            (
                obs,
                state,
                self.buf_rews[env_idx],
                terminated,
                truncated,
                self.buf_infos[env_idx],
            ) = self.envs[env_idx].step(self.actions[env_idx])
            self.buf_dones[env_idx] = terminated or truncated
            self.buf_infos[env_idx]['TimeLimit.truncated'] = (truncated and
                                                              not terminated)
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]['terminal_observation'] = obs
                obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()

            self._save_obs(env_idx, obs)
            self._save_state(env_idx, state)

        return (
            self._obs_from_buf(),
            self._state_from_buf(),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            deepcopy(self.buf_infos),
        )

    def reset(
        self
    ) -> Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]]:
        for env_idx in range(self.num_envs):
            maybe_options = ({
                'options': self._options[env_idx]
            } if self._options[env_idx] else {})
            obs, self.reset_infos[env_idx] = self.envs[env_idx].reset(
                seed=self._seeds[env_idx], **maybe_options)
            self._save_obs(env_idx, obs)
        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return self._obs_from_buf()

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        if self.render_mode != 'rgb_array':
            warnings.warn(
                f'The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images.'
            )
            return [None for _ in self.envs]
        return [env.render() for env in self.envs]  # type: ignore[misc]

    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        """Gym environment rendering. If there are multiple environments then
        they are tiled together in one image via ``BaseVecEnv.render()``.

        :param mode: The rendering type.
        """
        return super().render(mode=mode)

    def _save_obs(
        self,
        env_idx: int,
        obs: Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]],
    ) -> None:
        for key in self.obs_keys:
            if key is None:
                self.buf_obs[key][env_idx] = obs
            else:
                self.buf_obs[key][env_idx] = obs[
                    key]  # type: ignore[call-overload]

    def _save_state(
        self,
        env_idx: int,
        obs: Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]],
    ) -> None:
        for key in self.state_keys:
            if key is None:
                self.buf_state[key][env_idx] = obs
            else:
                self.buf_state[key][env_idx] = obs[
                    key]  # type: ignore[call-overload]

    def _obs_from_buf(
        self,
    ) -> Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]]:
        return dict_to_obs(self.obs_space, copy_obs_dict(self.buf_obs))

    def _state_from_buf(
        self,
    ) -> Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]]:
        return dict_to_obs(self.state_space, copy_obs_dict(self.buf_state))

    def get_attr(self,
                 attr_name: str,
                 indices: Union[None, int, Iterable[int]] = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(
        self,
        attr_name: str,
        value: Any,
        indices: Union[None, int, Iterable[int]] = None,
    ) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: Union[None, int, Iterable[int]] = None,
        **method_kwargs,
    ) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [
            getattr(env_i, method_name)(*method_args, **method_kwargs)
            for env_i in target_envs
        ]

    def env_is_wrapped(
        self,
        wrapper_class: Type[gym.Wrapper],
        indices: Union[None, int, Iterable[int]] = None,
    ) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper."""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        return [is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def _get_target_envs(
            self, indices: Union[None, int, Iterable[int]]) -> List[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]
