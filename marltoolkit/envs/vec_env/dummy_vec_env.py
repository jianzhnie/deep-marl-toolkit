import warnings
from copy import deepcopy
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Optional,
                    Sequence, Tuple, Type, Union)

import gymnasium as gym
import numpy as np
from gymnasium.vector.utils import concatenate, create_empty_array, iterate

from .base_vec_env import BaseVecEnv
from .utils import is_wrapped


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

    def __init__(
        self,
        env_fns: Iterator[Callable[[], gym.Env]],
        obs_space: gym.Space = None,
        state_space: gym.Space = None,
        action_space: gym.Space = None,
        copy: bool = True,
    ) -> None:
        self.env_fns = env_fns
        # Initialise all sub-environments
        self.envs = [fn() for fn in env_fns]
        # Define core attributes using the sub-environments
        # As we support `make_vec(spec)` then we can't include a `spec = self.envs[0].spec` as this doesn't guarantee we can actual recreate the vector env.
        self.num_envs = len(self.envs)
        self.metadata = self.envs[0].metadata
        self.render_mode = self.envs[0].render_mode

        if (obs_space is None) or (action_space is None):
            obs_space = obs_space or self.envs[0].obs_space
            state_space = state_space or self.envs[0].state_space
            action_space = action_space or self.envs[0].action_space

        super().__init__(
            num_envs=len(env_fns),
            obs_space=obs_space,
            state_space=state_space,
            action_space=action_space,
        )
        self._check_spaces()

        self._actions = None
        self.buf_obs = create_empty_array(self.single_obs_space,
                                          n=self.num_envs,
                                          fn=np.zeros)
        self.buf_state = create_empty_array(self.single_state_space,
                                            n=self.num_envs,
                                            fn=np.zeros)
        self.buf_rewards = np.zeros((self.num_envs, ), dtype=np.float32)
        self.buf_terminateds = np.zeros((self.num_envs, ), dtype=np.bool_)
        self.buf_truncateds = np.zeros((self.num_envs, ), dtype=np.bool_)
        self.buf_dones = np.zeros((self.num_envs, ), dtype=np.bool_)
        self.buf_infos: List[Dict[str,
                                  Any]] = [{} for _ in range(self.num_envs)]
        self.copy = copy
        self.metadata = self.envs[0].metadata

    def step_async(self, actions: Union[np.ndarray, List]) -> None:
        self._actions = iterate(self.action_space, actions)

    def step_wait(self):
        observations, states = [], []
        for env_idx, (env, action) in enumerate(zip(self.envs, self._actions)):
            (
                obs,
                state,
                reward,
                terminaled,
                truncated,
                info,
            ) = env.step(action)

            done = terminaled or truncated
            self.buf_dones[env_idx] = done
            self.buf_rewards[env_idx] = reward
            self.buf_terminateds[env_idx] = terminaled
            self.buf_truncateds[env_idx] = truncated
            self.buf_infos[env_idx]['info'] = info
            if done:
                old_obs, old_state, old_info = obs, state, info
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]['final_obs'] = old_obs
                self.buf_infos[env_idx]['final_state'] = old_state
                self.buf_infos[env_idx]['final_info'] = old_info
                obs, state, info = env.reset()

            observations.append(obs)
            states.append(state)

        self.buf_obs = concatenate(self.single_obs_space, observations,
                                   self.buf_obs)
        self.buf_state = concatenate(self.single_state_space, states,
                                     self.buf_state)

        return (
            deepcopy(self.buf_obs) if self.copy else self.buf_obs,
            deepcopy(self.buf_state) if self.copy else self.buf_state,
            np.copy(self.buf_rewards),
            np.copy(self.buf_dones),
            deepcopy(self.buf_infos),
        )

    def reset_wait(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ):
        """Waits for the calls triggered by :meth:`reset_async` to finish and
        returns the results.

        Args:
            seed: The reset environment seed
            options: Option information for the environment reset

        Returns:
            The reset observation of the environment and reset information
        """
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        self.buf_terminateds[:] = False
        self.buf_truncateds[:] = False
        self.buf_dones[:] = False
        observations, states = [], []

        for i, (env, single_seed) in enumerate(zip(self.envs, seed)):
            kwargs = {}
            if single_seed is not None:
                kwargs['seed'] = single_seed
            if options is not None:
                kwargs['options'] = options

            obs, state, info = env.reset(**kwargs)
            observations.append(obs)
            states.append(state)
            self.buf_infos[i] = info

        self.buf_obs = concatenate(self.single_obs_space, observations,
                                   self.buf_obs)
        self.buf_state = concatenate(self.single_obs_space, states,
                                     self.buf_state)
        return (
            (deepcopy(self.buf_obs) if self.copy else self.buf_obs),
            (deepcopy(self.buf_state) if self.copy else self.buf_state),
            (deepcopy(self.buf_infos) if self.copy else self.buf_infos),
        )

    def reset(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]]:
        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return self.reset_wait()

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

    def seed(self, seed: Optional[Union[int, Sequence[int]]] = None):
        """Sets the seed in all sub-environments.

        Args:
            seed: The seed
        """
        super().seed(seed=seed)
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        for env, single_seed in zip(self.envs, seed):
            env.seed(single_seed)

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

    def _check_spaces(self) -> bool:
        """Check that each of the environments obs and action spaces are
        equivalent to the single obs and action space."""
        for env in self.envs:
            if not (env.obs_space == self.single_obs_space):
                raise RuntimeError(
                    f'Some environments have an observation space different from `{self.single_obs_space}`. '
                    'In order to batch observations, the observation spaces from all environments must be equal.'
                )

            if not (env.action_space == self.single_action_space):
                raise RuntimeError(
                    f'Some environments have an action space different from `{self.single_action_space}`. '
                    'In order to batch actions, the action spaces from all environments must be equal.'
                )

        return True
