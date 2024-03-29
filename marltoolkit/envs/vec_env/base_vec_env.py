import inspect
from abc import ABC
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cloudpickle
import numpy as np
from gymnasium import logger, spaces
from gymnasium.vector.utils.spaces import batch_space

__all__ = ['BaseVecEnv', 'VecEnvWrapper', 'CloudpickleWrapper']


class BaseVecEnv(ABC):
    """Base class for vectorized environments to run multiple independent
    copies of the same environment in parallel.

    Vector environments can provide a linear speed-up in the steps taken per second through sampling multiple
    sub-environments at the same time. To prevent terminated environments waiting until all sub-environments have
    terminated or truncated, the vector environments autoreset sub-environments after they terminate or truncated.
    As a result, the final step's observation and info are overwritten by the reset's observation and info.
    Therefore, the observation and info for the final step of a sub-environment is stored in the info parameter,
    using `"final_observation"` and `"final_info"` respectively. See :meth:`step` for more information.

    The vector environments batch `observations`, `rewards`, `terminations`, `truncations` and `info` for each
    parallel environment. In addition, :meth:`step` expects to receive a batch of actions for each parallel environment.

    Gymnasium contains two types of Vector environments: :class:`AsyncVectorEnv` and :class:`SyncVectorEnv`.

    The Vector Environments have the additional attributes for users to understand the implementation

    - :attr:`num_envs` - The number of sub-environment in the vector environment
    - :attr:`obs_space` - The batched observation space of the vector environment
    - :attr:`state_space` - The batched state space of the vector environment
    - :attr:`single_obs_space` - The observation space of a single sub-environment
    - :attr:`single_state_space` - The state space of a single sub-environment
    - :attr:`action_space` - The batched action space of the vector environment
    - :attr:`single_action_space` - The action space of a single sub-environment
    - :attr:`closed (bool)` - Indicates whether the vectorized environment is closed.


    Note:
        The info parameter of :meth:`reset` and :meth:`step` was originally implemented before OpenAI Gym v25 was a list
        of dictionary for each sub-environment. However, this was modified in OpenAI Gym v25+ and in Gymnasium to a
        dictionary with a NumPy array for each key. To use the old info style using the :class:`VectorListInfo`.

    Note:
        To render the sub-environments, use :meth:`call` with "render" arguments. Remember to set the `render_modes`
        for all the sub-environments during initialization.

    Note:
        All parallel environments should share the identical observation and action spaces.
        In other words, a vector of multiple different environments is not supported.

    Attributes:
        num_envs (int): Number of environments in the vectorized environment.
        obs_space: The observation space of a single environment.
        action_space: The action space of a single environment.
    """

    def __init__(
        self,
        num_envs: int,
        obs_space: spaces.Space,
        state_space: spaces.Space,
        action_space: spaces.Space,
    ) -> None:
        self.num_envs = num_envs
        self.obs_space = batch_space(obs_space, n=num_envs)
        self.state_space = batch_space(state_space, n=num_envs)
        self.action_space = batch_space(action_space, n=num_envs)

        # The observation and action spaces of a single environment are
        # kept in separate properties
        self.single_obs_space = obs_space
        self.single_state_space = state_space
        self.single_action_space = action_space

        self.spec = None
        self.closed = False
        self.seeds = [None] * num_envs
        self.options = [{}] * num_envs

    def reset_async(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ):
        """Reset the sub-environments asynchronously.

        This method will return ``None``. A call to :meth:`reset_async` should be followed
        by a call to :meth:`reset_wait` to retrieve the results.

        Args:
            seed: The reset seed
            options: Reset options
        """
        pass

    def reset_wait(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ):
        """Retrieves the results of a :meth:`reset_async` call.

        A call to this method must always be preceded by a call to :meth:`reset_async`.

        Args:
            seed: The reset seed
            options: Reset options

        Returns:
            The results from :meth:`reset_async`

        Raises:
            NotImplementedError: VectorEnv does not implement function
        """
        raise NotImplementedError('VectorEnv does not implement function')

    def reset(
        self,
        *,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Reset all the environments and return an array of observations, or a
        dict of observation arrays.

        If step_async is still doing work, that work will be cancelled and
        step_wait() should not be called until step_async() is invoked again.

        Args:
            seed: The environment reset seeds
            options: If to return the options

        Returns:
            Union[np.ndarray, Dict[str, np.ndarray]]: Observations after reset.
        """
        self.reset_async(seed=seed, options=options)
        return self.reset_wait(seed=seed, options=options)

    def step_async(self, actions: Union[np.ndarray, List[Any]]) -> None:
        """Tell all the environments to start taking a step with the given
        actions.

        Call step_wait() to get the results of the step. You should not call
        this if a step_async run is already pending.

        Args:
            actions: Actions to take in each environment.
        """

    def step_wait(
        self,
    ) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], np.ndarray,
               np.ndarray, List[Dict[str, Any]], ]:
        """Wait for the step taken with step_async(). Returns (obs, rews,
        dones, infos):

        Args:
            actions: Actions to take in each environment.

        Returns:
            Tuple[Union[np.ndarray, Dict[str, np.ndarray]], np.ndarray, np.ndarray, List[Dict[str, Any]]]:
                - obs: Observations after the step.
                - rews: Rewards obtained from the step.
                - dones: "Episode done" booleans for each environment.
                - infos: A list of info objects for each environment.
        """
        raise NotImplementedError

    def step(self, actions: Union[np.ndarray, List[Any]]):
        """Perform a synchronous step: call step_async() and then wait for the
        results.

        Args:
            actions: Actions to take in each environment.

        Returns:
            Tuple[Union[np.ndarray, Dict[str, np.ndarray]], np.ndarray, np.ndarray, List[Dict[str, Any]]]:
                - obs: Observations after the step.
                - rews: Rewards obtained from the step.
                - dones: "Episode done" booleans for each environment.
                - infos: A list of info objects for each environment.
        """
        self.step_async(actions)
        return self.step_wait()

    def call_async(self, name, *args, **kwargs):
        """Calls a method name for each parallel environment asynchronously."""

    def call_wait(self, **kwargs) -> List[Any]:  # type: ignore
        """After calling a method in :meth:`call_async`, this function collects
        the results."""

    def call(self, name: str, *args, **kwargs) -> List[Any]:
        """Call a method, or get a property, from each parallel environment.

        Args:
            name (str): Name of the method or property to call.
            *args: Arguments to apply to the method call.
            **kwargs: Keyword arguments to apply to the method call.

        Returns:
            List of the results of the individual calls to the method or property for each environment.
        """
        self.call_async(name, *args, **kwargs)
        return self.call_wait()

    def close_extras(self, **kwargs: Any) -> None:
        """Clean up the extra resources, beyond what's in this base class.

        Only runs when not self.closed.
        """
        pass

    def close(self, **kwargs: Any) -> None:
        """Close the vectorized environment.

        If the environment is already closed, do nothing.
        """
        if self.closed:
            return

        self.close_extras(**kwargs)
        self.closed = True

    def _add_info(self, infos: dict, info: dict, env_num: int) -> dict:
        """Add env info to the info dictionary of the vectorized environment.

        Given the `info` of a single environment add it to the `infos` dictionary
        which represents all the infos of the vectorized environment.
        Every `key` of `info` is paired with a boolean mask `_key` representing
        whether or not the i-indexed environment has this `info`.

        Args:
            infos (dict): the infos of the vectorized environment
            info (dict): the info coming from the single environment
            env_num (int): the index of the single environment

        Returns:
            infos (dict): the (updated) infos of the vectorized environment
        """
        for k in info.keys():
            if k not in infos:
                info_array, array_mask = self._init_info_arrays(type(info[k]))
            else:
                info_array, array_mask = infos[k], infos[f'_{k}']

            info_array[env_num], array_mask[env_num] = info[k], True
            infos[k], infos[f'_{k}'] = info_array, array_mask
        return infos

    def _init_info_arrays(self, dtype: type) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize the info array.

        Initialize the info array. If the dtype is numeric
        the info array will have the same dtype, otherwise
        will be an array of `None`. Also, a boolean array
        of the same length is returned. It will be used for
        assessing which environment has info data.

        Args:
            dtype (type): data type of the info coming from the env.

        Returns:
            array (np.ndarray): the initialized info array.
            array_mask (np.ndarray): the initialized boolean array.
        """
        if dtype in [int, float, bool] or issubclass(dtype, np.number):
            array = np.zeros(self.num_envs, dtype=dtype)
        else:
            array = np.zeros(self.num_envs, dtype=object)
            array[:] = None
        array_mask = np.zeros(self.num_envs, dtype=bool)
        return array, array_mask

    def get_attr(self, name: str):
        """Get a property from each parallel environment.

        Args:
            name (str): Name of the property to be get from each individual environment.

        Returns:
            The property with name
        """
        return self.call(name)

    def set_attr(self, name: str, values: Union[list, tuple, object]):
        """Set a property in each sub-environment.

        Args:
            name (str): Name of the property to be set in each individual environment.
            values (list, tuple, or object): Values of the property to be set to. If `values` is a list or
                tuple, then it corresponds to the values for each individual environment, otherwise a single value
                is set for all environments.
        """

    def seed(self, seed: Optional[int] = None) -> Sequence[Union[None, int]]:
        """
        Sets the random seeds for all environments, based on a given seed.
        Each individual environment will still get its own seed, by incrementing the given seed.
        WARNING: since gym 0.26, those seeds will only be passed to the environment
        at the next reset.

        :param seed: The random seed. May be None for completely random seeding.
        :return: Returns a list containing the seeds for each individual env.
            Note that all list elements may be None, if the env does not return anything when being seeded.
        """
        if seed is None:
            # To ensure that subprocesses have different seeds,
            # we still populate the seed variable when no argument is passed
            seed = int(
                np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))

        self.seeds = [seed + idx for idx in range(self.num_envs)]
        return self.seeds

    def set_options(self,
                    options: Optional[Union[List[Dict], Dict]] = None) -> None:
        """
        Set environment options for all environments.
        If a dict is passed instead of a list, the same options will be used for all environments.
        WARNING: Those options will only be passed to the environment at the next reset.

        :param options: A dictionary of environment options to pass to each environment at the next reset.
        """
        if options is None:
            options = {}
        # Use deepcopy to avoid side effects
        if isinstance(options, dict):
            self.options = deepcopy([options] * self.num_envs)
        else:
            self.options = deepcopy(options)

    def getattr_depth_check(self, name: str,
                            already_found: bool) -> Optional[str]:
        """Check if an attribute reference is being hidden in a recursive call
        to __getattr__

        :param name: name of attribute to check for
        :param already_found: whether this attribute has already been found in a wrapper
        :return: name of module whose attribute is being shadowed, if any.
        """
        if hasattr(self, name) and already_found:
            return f'{type(self).__module__}.{type(self).__name__}'
        else:
            return None

    def __del__(self):
        """Closes the vector environment."""
        if not getattr(self, 'closed', True):
            self.close()

    def __repr__(self) -> str:
        """Returns a string representation of the vector environment.

        Returns:
            A string containing the class name, number of environments and environment spec id
        """
        if self.spec is None:
            return f'{self.__class__.__name__}({self.num_envs})'
        else:
            return f'{self.__class__.__name__}({self.spec.id}, {self.num_envs})'


class VecEnvWrapper(BaseVecEnv):
    """Vectorized environment base class.

    :param vec_env: the vectorized environment to wrap
    :param obs_space: the observation space (can be None to load from vec_env)
    :param action_space: the action space (can be None to load from vec_env)
    """

    def __init__(
        self,
        vec_env: BaseVecEnv,
        obs_space: Optional[spaces.Space] = None,
        state_space: Optional[spaces.Space] = None,
        action_space: Optional[spaces.Space] = None,
    ) -> None:
        self.vec_env = vec_env
        assert isinstance(vec_env, BaseVecEnv)

        super().__init__(
            num_envs=vec_env.num_envs,
            obs_space=obs_space or vec_env.obs_space,
            state_space=state_space or vec_env.state_space,
            action_space=action_space or vec_env.action_space,
        )
        self.class_attributes = dict(inspect.getmembers(self.__class__))

    def reset_async(self, **kwargs):
        return self.env.reset_async(**kwargs)

    def reset_wait(self, **kwargs):
        return self.env.reset_wait(**kwargs)

    def reset(self, **kwargs) -> None:
        return self.vec_env.reset(**kwargs)

    def step_async(self, actions: np.ndarray) -> None:
        return self.vec_env.step_async(actions)

    def step_wait(self) -> None:
        return self.vec_env.step_wait()

    def step(self, actions):
        """Step through all environments using the actions returning the
        batched data."""
        return self.vec_env.step(actions)

    def close_extras(self, **kwargs: Any):
        """Close all extra resources."""
        return self.vec_env.close_extras(**kwargs)

    def close(self, **kwargs: Any) -> None:
        return self.vec_env.close(**kwargs)

    def seed(self, seed: Optional[int] = None) -> Sequence[Union[None, int]]:
        return self.vec_env.seed(seed)

    def set_options(self,
                    options: Optional[Union[List[Dict], Dict]] = None) -> None:
        return self.vec_env.set_options(options)

    def get_attr(self, name: str) -> List[Any]:
        return self.vec_env.get_attr(name)

    def set_attr(self, name: str, values: Any) -> None:
        return self.vec_env.set_attr(name, values)

    def call(self, name, *args, **kwargs):
        return self.env.call(name, *args, **kwargs)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def num_envs(self) -> int:
        """Gets the wrapped vector environment's num of the sub-
        environments."""
        return self.vec_env.num_envs

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(
                f"attempted to get missing private attribute '{name}'")
        logger.warn(
            f'env.{name} to get variables from other wrappers is deprecated and will be removed in v1.0, '
            f'to get this variable you can do `env.unwrapped.{name}` for environment variables.'
        )
        return getattr(self.env, name)

    def __repr__(self):
        return f'<{self.__class__.__name__}, {self.env}>'

    def __del__(self):
        self.env.__del__()


class CloudpickleWrapper:
    """Wrapper class that uses cloudpickle to serialize contents, enabling
    proper serialization for multiprocessing."""

    def __init__(self, x) -> None:
        """Initialize the CloudpickleWrapper with the given object.

        Args:
            x: The object to be wrapped and serialized.
        """
        self.x = x

    def __getstate__(self) -> bytes:
        """Get the state of the object for serialization using cloudpickle.

        Returns:
            bytes: Serialized representation of the object using cloudpickle.
        """
        return cloudpickle.dumps(self.x)

    def __setstate__(self, x: Any) -> None:
        """Set the state of the object by deserializing using pickle.

        Args:
            ob: Serialized representation of the object.
        """

        self.x = cloudpickle.loads(x)
