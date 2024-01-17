from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cloudpickle
import numpy as np
from gymnasium import spaces


class VecEnv(ABC):
    """Abstract base class for vectorized environments.

    This class defines the interface for interacting with a vectorized environment,
    where multiple environments can be stepped in parallel.

    Attributes:
        num_envs (int): Number of environments in the vectorized environment.
        observation_space: The observation space of a single environment.
        action_space: The action space of a single environment.
        closed (bool): Indicates whether the vectorized environment is closed.
    """

    closed: bool = False
    viewer: Any = None

    metadata: Dict[str, List[str]] = {'render.modes': ['human', 'rgb_array']}

    def __init__(
        self,
        num_envs: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
    ):
        """Initialize the vectorized environment.

        Args:
            num_envs (int): Number of environments in the vectorized environment.
            observation_space: The observation space of a single environment.
            action_space: The action space of a single environment.
        """
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
        # store info returned by the reset method
        self.reset_infos: List[Dict[str, Any]] = [{} for _ in range(num_envs)]
        # seeds to be used in the next call to env.reset()
        self._seeds: List[Optional[int]] = [None for _ in range(num_envs)]
        # options to be used in the next call to env.reset()
        self._options: List[Dict[str, Any]] = [{} for _ in range(num_envs)]
        self.closed = False
        self.render_mode = None

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
        raise NotImplementedError()

    @abstractmethod
    def step_async(self, actions: Union[np.ndarray, List[Any]]) -> None:
        """Tell all the environments to start taking a step with the given
        actions.

        Call step_wait() to get the results of the step. You should not call
        this if a step_async run is already pending.

        Args:
            actions: Actions to take in each environment.
        """
        raise NotImplementedError()

    @abstractmethod
    def step_wait(
        self
    ) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], np.ndarray,
               np.ndarray, List[Dict[str, Any]]]:
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
        raise NotImplementedError()

    def step(
        self, actions: Union[np.ndarray, List[Any]]
    ) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], np.ndarray,
               np.ndarray, List[Dict[str, Any]]]:
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

    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        """Gym environment rendering.

        :param mode: the rendering type
        """
        raise NotImplementedError()

    def get_images(self) -> List[np.ndarray]:
        """Return RGB images from each environment.

        Returns:
            List[np.ndarray]: List of RGB images.
        """
        raise NotImplementedError()

    @abstractmethod
    def close_extras(self) -> None:
        """Clean up the extra resources, beyond what's in this base class.

        Only runs when not self.closed.
        """
        raise NotImplementedError()

    def close(self) -> None:
        """Close the vectorized environment.

        If the environment is already closed, do nothing.
        """
        if self.closed:
            return
        self.close_extras()
        self.closed = True

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


class CloudpickleWrapper:
    """Wrapper class that uses cloudpickle to serialize contents, enabling
    proper serialization for multiprocessing."""

    def __init__(self, x):
        """Initialize the CloudpickleWrapper with the given object.

        Args:
            x: The object to be wrapped and serialized.
        """
        self.x = x

    def __getstate__(self):
        """Get the state of the object for serialization using cloudpickle.

        Returns:
            bytes: Serialized representation of the object using cloudpickle.
        """
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        """Set the state of the object by deserializing using pickle.

        Args:
            ob: Serialized representation of the object.
        """
        self.x = cloudpickle.loads(ob)
