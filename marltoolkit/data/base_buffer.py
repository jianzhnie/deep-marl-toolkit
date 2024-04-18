from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
import torch


class BaseBuffer(ABC):
    """Abstract base class for all buffers.

    This class provides a basic structure for reinforcement learning experience
    replay buffers.

    Args:
        :param num_envs: Number of environments.
        :param num_agents: Number of agents.
        :param obs_shape: Dimensionality of the observation space.
        :param state_shape: Dimensionality of the state space.
        :param action_shape: Dimensionality of the action space.
        :param reward_shape: Dimensionality of the reward space.
        :param done_shape: Dimensionality of the done space.
        :param device: Device on which to store the buffer data.
        :param kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        num_envs: int,
        num_agents: int,
        obs_shape: Union[int, Tuple],
        state_shape: Union[int, Tuple],
        action_shape: Union[int, Tuple],
        reward_shape: Union[int, Tuple],
        done_shape: Union[int, Tuple],
        device: Union[torch.device, str] = 'cpu',
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.obs_shape = obs_shape
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.reward_shape = reward_shape
        self.done_shape = done_shape
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

    @abstractmethod
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
