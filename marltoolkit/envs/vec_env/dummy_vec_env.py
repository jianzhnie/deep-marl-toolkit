from copy import deepcopy
from typing import Any, Callable, Iterator, List, Optional, Sequence, Union

import gymnasium as gym
import numpy as np
from gymnasium.vector.utils import concatenate, create_empty_array, iterate

from .base_vec_env import BaseVecEnv

__all__ = ['DummyVecEnv']


class DummyVecEnv(BaseVecEnv):
    """Creates a simple vectorized wrapper for multiple environments, calling
    each environment in sequence on the current Python process. This is useful
    for computationally simple environment such as ``Cartpole-v1``, as the
    overhead of multiprocess or multithread outweighs the environment
    computation time. This can also be used for RL methods that require a
    vectorized environment, but that you want a single environments to train
    with.

    Args:
        env_fns: iterable of callable functions that create the environments.
        observation_space: Observation space of a single environment. If ``None``,
            then the observation space of the first environment is taken.
        state_space: State space of a single environment. If ``None``,
            then the state space of the first environment is taken.
        action_space: Action space of a single environment. If ``None``,
            then the action space of the first environment is taken.
        copy: If ``True``, then the :meth:`reset` and :meth:`step` methods return a copy of the observations.

    Raises:
        RuntimeError: If the observation space of some sub-environment does not match observation_space
            (or, by default, the observation space of the first sub-environment).
    """

    def __init__(
        self,
        env_fns: Iterator[Callable[[], gym.Env]],
        obs_space: gym.Space = None,
        state_space: gym.Space = None,
        action_space: gym.Space = None,
        copy: bool = True,
    ) -> None:
        # Initialise all sub-environments
        self.envs = [env_fn() for env_fn in env_fns]
        # Define core attributes using the sub-environments
        # As we support `make_vec(spec)` then we can't include a `spec = self.envs[0].spec` as this doesn't guarantee we can actual recreate the vector env.
        self.num_envs = len(self.envs)

        if (obs_space is None) or (state_space is None) or (action_space is
                                                            None):
            obs_space = obs_space or self.envs[0].obs_space
            state_space = state_space or self.envs[0].state_space
            action_space = action_space or self.envs[0].action_space

        super().__init__(
            num_envs=self.num_envs,
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
        self.buf_dones = np.zeros((self.num_envs, ), dtype=np.bool_)
        self.copy = copy

    def seed(self, seed: Optional[Union[int, Sequence[int]]] = None):
        """Sets the seed in all sub-environments.

        Args:
            seed: The seed
        """
        super().seed(seed=seed)
        if seed is None:
            self.seeds = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            self.seeds = [seed + i for i in range(self.num_envs)]
        assert len(self.seeds) == self.num_envs

        for env, single_seed in zip(self.envs, self.seeds):
            env.seed(single_seed)

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
            self.seeds = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            self.seeds = [seed + i for i in range(self.num_envs)]
        assert len(self.seeds) == self.num_envs

        self.buf_dones[:] = False
        observations, states, infos = ([], [], {})

        for env_idx, env in enumerate(self.envs):
            kwargs = {}
            single_seed = self.seeds[env_idx]
            if single_seed is not None:
                kwargs['seed'] = single_seed
            if options is not None:
                kwargs['options'] = options

            obs, state, info = env.reset(**kwargs)
            observations.append(obs)
            states.append(state)
            infos = self._add_info(infos, info, env_idx)

        self.buf_obs = concatenate(self.single_obs_space, observations,
                                   self.buf_obs)
        self.buf_state = concatenate(self.single_obs_space, states,
                                     self.buf_state)
        return (
            (deepcopy(self.buf_obs) if self.copy else self.buf_obs),
            (deepcopy(self.buf_state) if self.copy else self.buf_state),
            (deepcopy(infos) if self.copy else infos),
        )

    def step_async(self, actions: Union[np.ndarray, List]) -> None:
        """Sets :attr:`_actions` for use by the :meth:`step_wait` by converting
        the ``actions`` to an iterable version."""
        self._actions = iterate(self.action_space, actions)

    def step_wait(self):
        """Steps through each of the environments returning the batched
        results.

        Returns:
            The batched environment step results
        """
        observations, states, infos = [], [], {}
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
            if done:
                old_obs, old_state, old_info = obs, state, info
                # save final observation where user can get it, then reset
                info['final_obs'] = old_obs
                info['final_state'] = old_state
                info['final_info'] = old_info
                obs, state, info = env.reset()

            observations.append(obs)
            states.append(state)
            infos = self._add_info(infos, info, env_idx)

        self.buf_obs = concatenate(self.single_obs_space, observations,
                                   self.buf_obs)
        self.buf_state = concatenate(self.single_state_space, states,
                                     self.buf_state)

        return (
            deepcopy(self.buf_obs) if self.copy else self.buf_obs,
            deepcopy(self.buf_state) if self.copy else self.buf_state,
            np.copy(self.buf_rewards),
            np.copy(self.buf_dones),
            infos,
        )

    def call(self, name, *args, **kwargs) -> tuple:
        """Calls the method with name and applies args and kwargs.

        Args:
            name: The method name
            *args: The method args
            **kwargs: The method kwargs

        Returns:
            Tuple of results
        """
        results = []
        for env in self.envs:
            function = getattr(env, name)
            if callable(function):
                results.append(function(*args, **kwargs))
            else:
                results.append(function)

        return tuple(results)

    def close_extras(self, **kwargs):
        """Close the environments."""
        for env in self.envs:
            env.close()

    def set_attr(self, name: str, values: Union[list, tuple, Any]) -> None:
        """Sets an attribute of the sub-environments.

        Args:
            name: The property name to change
            values: Values of the property to be set to. If ``values`` is a list or
                tuple, then it corresponds to the values for each individual
                environment, otherwise, a single value is set for all environments.

        Raises:
            ValueError: Values must be a list or tuple with length equal to the number of environments.
        """
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]
        if len(values) != self.num_envs:
            raise ValueError(
                'Values must be a list or tuple with length equal to the '
                f'number of environments. Got `{len(values)}` values for '
                f'{self.num_envs} environments.')
        for env, value in zip(self.envs, values):
            setattr(env, name, value)

    def _check_spaces(self) -> bool:
        """Check that each of the environments obs and action spaces are
        equivalent to the single obs and action space."""
        for env in self.envs:
            if not (env.obs_space == self.single_obs_space):
                raise RuntimeError(
                    f'Some environments have an observation space different from `{self.single_obs_space}`. '
                    'In order to batch observations, the observation spaces from all environments must be equal.'
                )

            if not (env.state_space == self.single_state_space):
                raise RuntimeError(
                    f'Some environments have an state space different from `{self.single_state_space}`. '
                    'In order to batch state, the state spaces from all environments must be equal.'
                )

            if not (env.action_space == self.single_action_space):
                raise RuntimeError(
                    f'Some environments have an action space different from `{self.single_action_space}`. '
                    'In order to batch actions, the action spaces from all environments must be equal.'
                )

        return True
