import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cloudpickle
import numpy as np
from gymnasium import spaces

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
        share_obs_space: spaces.Space,
        action_space: spaces.Space,
    ) -> None:
        self.closed = False
        self.num_envs = num_envs
        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.action_space = action_space
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
                # SB3 uses OpenCV for the "human" mode
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

    @abstractmethod
    def reset(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Reset all the environments and return an array of observations, or a
        dict of observation arrays.

        If step_async is still doing work, that work will be cancelled and
        step_wait() should not be called until step_async() is invoked again.

        Returns:
            Union[np.ndarray, Dict[str, np.ndarray]]: Observations after reset.
        """
        raise NotImplementedError

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

    def close_extras(self) -> None:
        """Clean up the extra resources, beyond what's in this base class.

        Only runs when not self.closed.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Close the vectorized environment.

        If the environment is already closed, do nothing.
        """
        if self.closed:
            return
        self.close_extras()
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
        raise NotImplementedError()

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
        raise NotImplementedError()

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
        raise NotImplementedError()

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
