import warnings
from copy import deepcopy
from typing import Any, Callable, Iterator, List, Optional, Sequence, Union

import gymnasium as gym
import numpy as np
from gymnasium.vector.utils import concatenate, create_empty_array, iterate

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

        self.copy = copy
        self.metadata = self.envs[0].metadata

    def seed(self, seed: Optional[Union[int, Sequence[int]]] = None):
        """Sets the seed in all sub-environments.

        Args:
            seed: The seed
        """
        super().seed(seed=seed)
        if seed is None:
            self.seeds_ = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            self._seeds = [seed + i for i in range(self.num_envs)]
        assert len(self._seeds) == self.num_envs

        for env, single_seed in zip(self.envs, self._seeds):
            env.seed(single_seed)

    def step_async(self, actions: Union[np.ndarray, List]) -> None:
        self._actions = iterate(self.action_space, actions)

    def step_wait(self):
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
            self.buf_terminateds[env_idx] = terminaled
            self.buf_truncateds[env_idx] = truncated
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

    def reset_wait(self):
        """Waits for the calls triggered by :meth:`reset_async` to finish and
        returns the results.

        Args:
            seed: The reset environment seed
            options: Option information for the environment reset

        Returns:
            The reset observation of the environment and reset information
        """
        self.buf_terminateds[:] = False
        self.buf_truncateds[:] = False
        self.buf_dones[:] = False

        observations, states, reset_infos = ([], [], {})

        for env_idx, env in enumerate(self.envs):
            single_seed = self._seeds[env_idx]
            maybe_options = ({
                'options': self._options[env_idx]
            } if self._options[env_idx] else {})

            obs, state, info = env.reset(seed=single_seed, **maybe_options)
            observations.append(obs)
            states.append(state)
            reset_infos = self._add_info(reset_infos, info, env_idx)

        self.buf_obs = concatenate(self.single_obs_space, observations,
                                   self.buf_obs)
        self.buf_state = concatenate(self.single_obs_space, states,
                                     self.buf_state)
        return (
            (deepcopy(self.buf_obs) if self.copy else self.buf_obs),
            (deepcopy(self.buf_state) if self.copy else self.buf_state),
            (deepcopy(reset_infos) if self.copy else reset_infos),
        )

    def reset(self):
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

    def get_attr(self, attr_name: str) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        return [getattr(env_i, attr_name) for env_i in self.envs]

    def set_attr(self, attr_name: str, values: Any) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]
        if len(values) != self.num_envs:
            raise ValueError(
                'Values must be a list or tuple with length equal to the '
                f'number of environments. Got `{len(values)}` values for '
                f'{self.num_envs} environments.')
        for env, value in zip(self.envs, values):
            setattr(env, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args,
        **method_kwargs,
    ) -> List[Any]:
        """Call instance methods of vectorized environments."""
        return [
            getattr(env_i, method_name)(*method_args, **method_kwargs)
            for env_i in self.envs
        ]

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
