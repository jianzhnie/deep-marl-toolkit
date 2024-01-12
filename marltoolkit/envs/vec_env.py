from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import numpy as np


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

    def __init__(self, num_envs: int, observation_space: Any,
                 action_space: Any):
        """Initialize the vectorized environment.

        Args:
            num_envs (int): Number of environments in the vectorized environment.
            observation_space: The observation space of a single environment.
            action_space: The action space of a single environment.
        """
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
        self.closed = False

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
        raise NotImplementedError

    @abstractmethod
    def close_extras(self) -> None:
        """Clean up the extra resources, beyond what's in this base class.

        Only runs when not self.closed.
        """
        raise NotImplementedError

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

    def render(self, mode: str) -> None:
        """Render the vectorized environment.

        Args:
            mode (str): Rendering mode.
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
