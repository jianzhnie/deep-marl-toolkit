from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
import torch


class BaseBuffer(ABC):
    """Abstract base class for all buffers.

    This class provides a basic structure for reinforcement learning experience
    replay buffers.
    """

    def __init__(
        self,
        max_size: int,
        num_envs: int,
        num_agents: int,
        obs_space: Union[int, Tuple],
        state_space: Union[int, Tuple],
        action_space: Union[int, Tuple],
        reward_space: Union[int, Tuple],
        done_space: Union[int, Tuple],
        device: Union[torch.device, str] = 'cpu',
        **kwargs,
    ):
        """Initialize the base buffer.

        :param num_envs: Number of environments.
        :param max_size: Maximum capacity of the buffer.
        :param num_agents: Number of agents.
        :param state_space: Dimensionality of the state space.
        :param obs_space: Dimensionality of the observation space.
        :param action_space: Dimensionality of the action space.
        :param reward_space: Dimensionality of the reward space.
        :param done_space: Dimensionality of the done space.
        :param device: Device on which to store the buffer data.
        :param kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.num_envs = num_envs
        self.max_size = max_size
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.state_space = state_space
        self.action_space = action_space
        self.reward_space = reward_space
        self.done_space = done_space
        self.device = device
        self.curr_ptr = 0
        self.curr_size = 0

    def reset(self) -> None:
        """Reset the buffer."""
        self.curr_ptr = 0
        self.curr_size = 0

    @abstractmethod
    def store(self, *args) -> None:
        """Add elements to the buffer."""
        raise NotImplementedError

    def extend(self, *args, **kwargs) -> None:
        """Add a new batch of transitions to the buffer."""
        for data in zip(*args):
            self.store(*data)

    @abstractmethod
    def sample(self, **kwargs):
        """Sample elements from the buffer."""
        raise NotImplementedError

    @abstractmethod
    def store_transitions(self, **kwargs):
        """Store transitions in the buffer."""
        raise NotImplementedError

    @abstractmethod
    def store_episodes(self, **kwargs):
        """Store episodes in the buffer."""
        raise NotImplementedError

    @abstractmethod
    def finish_path(self, **kwargs):
        """Finish a trajectory path in the buffer."""
        raise NotImplementedError

    def size(self) -> int:
        """Get the current size of the buffer.

        :return: The current size of the buffer.
        """
        return self.curr_size

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """Convert a numpy array to a PyTorch tensor.

        :param array: Numpy array to be converted.
        :param copy: Whether to copy the data or not.
        :return: PyTorch tensor.
        """
        if copy:
            return torch.tensor(array).to(self.device)
        return torch.as_tensor(array).to(self.device)
