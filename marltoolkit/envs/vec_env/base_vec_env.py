import inspect
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import (Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type,
                    Union)

import cloudpickle
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils.spaces import batch_space

from marltoolkit.utils.util import tile_images


class BaseVecEnv(ABC):
    """Abstract base class for vectorized environments.

    This class defines the interface for interacting with a vectorized environment,
    where multiple environments can be stepped in parallel.

    Attributes:
        num_envs (int): Number of environments in the vectorized environment.
        obs_space: The observation space of a single environment.
        action_space: The action space of a single environment.
        closed (bool): Indicates whether the vectorized environment is closed.
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

        self.is_vector_env = True
        self.closed = False

        # The observation and action spaces of a single environment are
        # kept in separate properties
        self.single_obs_space = obs_space
        self.single_state_space = state_space
        self.single_action_space = action_space

        # store info returned by the reset method
        self.reset_infos: List[Dict[str, Any]] = [{} for _ in range(num_envs)]
        # seeds to be used in the next call to env.reset()
        self._seeds: List[Optional[int]] = [None for _ in range(num_envs)]
        # options to be used in the next call to env.reset()
        self._options: List[Dict[str, Any]] = [{} for _ in range(num_envs)]

        try:
            render_modes = self.get_attr('render_mode')
        except AttributeError:
            warnings.warn(
                'The `render_mode` attribute is not defined in your environment. It will be set to None.'
            )
            render_modes = [None for _ in range(num_envs)]

        assert all(
            render_mode == render_modes[0] for render_mode in render_modes
        ), 'render_mode mode should be the same for all environments'
        self.render_mode = render_modes[0]

        render_modes = []
        if self.render_mode is not None:
            if self.render_mode == 'rgb_array':
                render_modes = ['human', 'rgb_array']
            else:
                render_modes = [self.render_mode]

        self.metadata = {'render_modes': render_modes}

    def _reset_seeds(self) -> None:
        """Reset the seeds that are going to be used at the next reset."""
        self._seeds = [None for _ in range(self.num_envs)]

    def _reset_options(self) -> None:
        """Reset the options that are going to be used at the next reset."""
        self._options = [{} for _ in range(self.num_envs)]

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

    @abstractmethod
    def reset(
            self, seed: int,
            options: dict[str,
                          Any]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Reset all the environments and return an array of observations, or a
        dict of observation arrays.

        If step_async is still doing work, that work will be cancelled and
        step_wait() should not be called until step_async() is invoked again.

        Returns:
            Union[np.ndarray, Dict[str, np.ndarray]]: Observations after reset.
        """
        self.reset_async(seed=seed, options=options)
        return self.reset_wait(seed=seed, options=options)

    @abstractmethod
    def step_async(self, actions: Union[np.ndarray, List[Any]]) -> None:
        """Tell all the environments to start taking a step with the given
        actions.

        Call step_wait() to get the results of the step. You should not call
        this if a step_async run is already pending.

        Args:
            actions: Actions to take in each environment.
        """
        raise NotImplementedError

    @abstractmethod
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

    def close_extras(self, **kwargs: Any) -> None:
        """Clean up the extra resources, beyond what's in this base class.

        Only runs when not self.closed.
        """
        raise NotImplementedError

    def close(self, **kwargs: Any) -> None:
        """Close the vectorized environment.

        If the environment is already closed, do nothing.
        """
        if self.closed:
            return

        self.close_extras(**kwargs)
        self.closed = True

    @abstractmethod
    def get_attr(self,
                 attr_name: str,
                 indices: Union[None, int, Iterable[int]] = None) -> List[Any]:
        """Return attribute from vectorized environment.

        :param attr_name: The name of the attribute whose value to return
        :param indices: Indices of envs to get attribute from
        :return: List of values of 'attr_name' in all environments
        """
        raise NotImplementedError

    @abstractmethod
    def set_attr(
        self,
        attr_name: str,
        value: Any,
        indices: Union[None, int, Iterable[int]] = None,
    ) -> None:
        """Set attribute inside vectorized environments.

        :param attr_name: The name of attribute to assign new value
        :param value: Value to assign to `attr_name`
        :param indices: Indices of envs to assign value
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: Union[None, int, Iterable[int]] = None,
        **method_kwargs,
    ) -> List[Any]:
        """Call instance methods of vectorized environments.

        :param method_name: The name of the environment method to invoke.
        :param indices: Indices of envs whose method to call
        :param method_args: Any positional arguments to provide in the call
        :param method_kwargs: Any keyword arguments to provide in the call
        :return: List of items returned by the environment's method call
        """
        raise NotImplementedError

    def env_is_wrapped(
        self,
        wrapper_class: Type[gym.Wrapper],
        indices: Union[None, int, Iterable[int]] = None,
    ) -> List[bool]:
        """Check if environments are wrapped with a given wrapper.

        :param method_name: The name of the environment method to invoke.
        :param indices: Indices of envs whose method to call
        :param method_args: Any positional arguments to provide in the call
        :param method_kwargs: Any keyword arguments to provide in the call
        :return: True if the env is wrapped, False otherwise, for each env queried.
        """
        raise NotImplementedError

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        """Return RGB images from each environment.

        Returns:
            List[np.ndarray]: List of RGB images.
        """
        raise NotImplementedError

    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        """Gym environment rendering.

        :param mode: the rendering type
        """

        if mode == 'human' and self.render_mode != mode:
            # Special case, if the render_mode="rgb_array"
            # we can still display that image using opencv
            if self.render_mode != 'rgb_array':
                warnings.warn(
                    f"You tried to render a VecEnv with mode='{mode}' "
                    'but the render mode defined when initializing the environment must be '
                    f"'human' or 'rgb_array', not '{self.render_mode}'.")
                return None

        elif mode and self.render_mode != mode:
            warnings.warn(
                f"""Starting from gymnasium v0.26, render modes are determined during the initialization of the environment.
                We allow to pass a mode argument to maintain a backwards compatible VecEnv API, but the mode ({mode})
                has to be the same as the environment render mode ({self.render_mode}) which is not the case."""
            )
            return None

        mode = mode or self.render_mode

        if mode is None:
            warnings.warn(
                'You tried to call render() but no `render_mode` was passed to the env constructor.'
            )
            return None

        # mode == self.render_mode == "human"
        # In that case, we try to call `self.env.render()` but it might
        # crash for subprocesses
        if self.render_mode == 'human':
            self.env_method('render')
            return None

        if mode == 'rgb_array' or mode == 'human':
            # call the render method of the environments
            images = self.get_images()
            # Create a big image by tiling images from subprocesses
            bigimg = tile_images(images)  # type: ignore[arg-type]

            if mode == 'human':
                # Display it using OpenCV
                import cv2

                cv2.imshow('vecenv', bigimg[:, :, ::-1])
                cv2.waitKey(1)
            else:
                return bigimg

        else:
            # Other render modes:
            # In that case, we try to call `self.env.render()` but it might
            # crash for subprocesses
            # and we don't return the values
            self.env_method('render')
        return None

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

        self._seeds = [seed + idx for idx in range(self.num_envs)]
        return self._seeds

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
            self._options = deepcopy([options] * self.num_envs)
        else:
            self._options = deepcopy(options)

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

    def _get_indices(
            self, indices: Union[None, int, Iterable[int]]) -> Iterable[int]:
        """Convert a flexibly-typed reference to environment indices to an
        implied list of indices.

        :param indices: refers to indices of envs.
        :return: the implied list of indices.
        """
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
        return indices


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

    @abstractmethod
    def reset(self, seed: int, options: dict[str, Any]) -> None:
        return self.vec_env.reset(seed=seed, options=options)

    def step_async(self, actions: np.ndarray) -> None:
        return self.vec_env.step_async(actions)

    @abstractmethod
    def step_wait(self) -> None:
        return self.vec_env.step_wait()

    def step(self, actions):
        """Step through all environments using the actions returning the
        batched data."""
        return self.vec_env.step(actions)

    def seed(self, seed: Optional[int] = None) -> Sequence[Union[None, int]]:
        return self.vec_env.seed(seed)

    def set_options(self,
                    options: Optional[Union[List[Dict], Dict]] = None) -> None:
        return self.vec_env.set_options(options)

    def close_extras(self, **kwargs: Any):
        """Close all extra resources."""
        return self.vec_env.close_extras(**kwargs)

    def close(self, **kwargs: Any) -> None:
        return self.vec_env.close(**kwargs)

    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        return self.vec_env.render(mode=mode)

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        return self.vec_env.get_images()

    def get_attr(self,
                 attr_name: str,
                 indices: Union[None, int, Iterable[int]] = None) -> List[Any]:
        return self.vec_env.get_attr(attr_name, indices)

    def set_attr(
        self,
        attr_name: str,
        value: Any,
        indices: Union[None, int, Iterable[int]] = None,
    ) -> None:
        return self.vec_env.set_attr(attr_name, value, indices)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: Union[None, int, Iterable[int]] = None,
        **method_kwargs,
    ) -> List[Any]:
        return self.vec_env.env_method(method_name,
                                       *method_args,
                                       indices=indices,
                                       **method_kwargs)

    def env_is_wrapped(
        self,
        wrapper_class: Type[gym.Wrapper],
        indices: Union[None, int, Iterable[int]] = None,
    ) -> List[bool]:
        return self.vec_env.env_is_wrapped(wrapper_class, indices=indices)

    @property
    def num_envs(self) -> int:
        """Gets the wrapped vector environment's num of the sub-
        environments."""
        return self.vec_env.num_envs

    @property
    def render_mode(self):
        """Returns the `render_mode` from the base environment."""
        return self.vec_env.render_mode

    def __getattr__(self, name: str) -> Any:
        """Find attribute from wrapped vec_env(s) if this wrapper does not have
        it.

        Useful for accessing attributes from vec_envs which are wrapped with
        multiple wrappers which have unique attributes of interest.
        """
        blocked_class = self.getattr_depth_check(name, already_found=False)
        if blocked_class is not None:
            own_class = f'{type(self).__module__}.{type(self).__name__}'
            error_str = (
                f'Error: Recursive attribute lookup for {name} from {own_class} is '
                f'ambiguous and hides attribute from {blocked_class}')
            raise AttributeError(error_str)

        return self.getattr_recursive(name)

    def _get_all_attributes(self) -> Dict[str, Any]:
        """Get all (inherited) instance and class attributes.

        :return: all_attributes
        """
        all_attributes = self.__dict__.copy()
        all_attributes.update(self.class_attributes)
        return all_attributes

    def getattr_recursive(self, name: str) -> Any:
        """Recursively check wrappers to find attribute.

        :param name: name of attribute to look for
        :return: attribute
        """
        all_attributes = self._get_all_attributes()
        if name in all_attributes:  # attribute is present in this wrapper
            attr = getattr(self, name)
        elif hasattr(self.vec_env, 'getattr_recursive'):
            # Attribute not present, child is wrapper. Call getattr_recursive rather than getattr
            # to avoid a duplicate call to getattr_depth_check.
            attr = self.vec_env.getattr_recursive(name)
        else:  # attribute not present, child is an unwrapped VecEnv
            attr = getattr(self.vec_env, name)

        return attr

    def getattr_depth_check(self, name: str,
                            already_found: bool) -> Optional[str]:
        """See base class.

        :return: name of module whose attribute is being shadowed, if any.
        """
        all_attributes = self._get_all_attributes()
        if name in all_attributes and already_found:
            # this vec_env's attribute is being hidden because of a higher vec_env.
            shadowed_wrapper_class: Optional[str] = (
                f'{type(self).__module__}.{type(self).__name__}')
        elif name in all_attributes and not already_found:
            # we have found the first reference to the attribute. Now check for duplicates.
            shadowed_wrapper_class = self.vec_env.getattr_depth_check(
                name, True)
        else:
            # this wrapper does not have the attribute. Keep searching.
            shadowed_wrapper_class = self.vec_env.getattr_depth_check(
                name, already_found)

        return shadowed_wrapper_class


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
