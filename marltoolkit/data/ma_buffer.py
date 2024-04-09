from typing import Dict, Tuple, Union

import numpy as np
import torch

__all__ = ['EpisodeData', 'ReplayBuffer']


class EpisodeData:
    """Class for managing and storing episode data."""

    def __init__(
        self,
        num_agents: int,
        num_actions: int,
        episode_limit: int,
        obs_shape: Union[int, Tuple],
        state_shape: Union[int, Tuple],
    ):
        """Initialize EpisodeData.

        Parameters:
        - num_agents (int): Number of agents.
        - num_actions (int): Number of possible actions.
        - episode_limit (int): Maximum number of steps in an episode.
        - obs_shape (Union[int, Tuple]): Shape of the observation.
        - state_shape (Union[int, Tuple]): Shape of the state.
        """
        self.obs_buf = np.zeros((episode_limit, num_agents, obs_shape))
        self.state_buf = np.zeros((episode_limit, state_shape))
        self.action_buf = np.zeros((episode_limit, num_agents))
        self.action_onehot_buf = np.zeros(
            (episode_limit, num_agents, num_actions))
        self.available_action_buf = np.zeros(
            (episode_limit, num_agents, num_actions))
        self.reward_buf = np.zeros((episode_limit, 1))
        self.terminal_buf = np.zeros((episode_limit, 1))
        self.filled_buf = np.zeros((episode_limit, 1))

        # memory management
        self.curr_ptr = 0
        self.curr_size = 0
        self.episode_limit = episode_limit

    def add(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        actions_onehot: np.ndarray,
        available_actions: np.ndarray,
        rewards: np.ndarray,
        terminated: np.ndarray,
        filled: np.ndarray,
    ) -> None:
        """Add a step of data to the episode.

        Parameters:
        - obs (np.ndarray): Observation data.
        - state (np.ndarray): State data.
        - actions (np.ndarray): Actions taken.
        - actions_onehot (np.ndarray): Actions in one-hot encoding.
        - available_actions (np.ndarray): Available actions for each agent.
        - rewards (np.ndarray): Reward received.
        - terminated (np.ndarray): Termination flag.
        - filled (np.ndarray): Filled flag.
        """
        assert self.size() < self.episode_limit
        self.obs_buf[self.curr_ptr] = obs
        self.state_buf[self.curr_ptr] = state
        self.action_buf[self.curr_ptr] = actions
        self.action_onehot_buf[self.curr_ptr] = actions_onehot
        self.available_action_buf[self.curr_ptr] = available_actions

        self.reward_buf[self.curr_ptr] = rewards
        self.terminal_buf[self.curr_ptr] = terminated
        self.filled_buf[self.curr_ptr] = filled
        self.curr_ptr += 1
        self.curr_size = min(self.curr_size + 1, self.episode_limit)

    def fill_mask(self) -> None:
        """Fill the mask for the current step."""
        assert self.size() < self.episode_limit
        self.terminal_buf[self.curr_ptr] = True
        self.filled_buf[self.curr_ptr] = 1.0
        self.curr_ptr += 1
        self.curr_size = min(self.curr_size + 1, self.episode_limit)

    def get_data(self) -> Dict[str, np.ndarray]:
        """Get all the data in an episode.

        Returns:
        - Dict[str, np.ndarray]: Episode data dictionary.
        """
        assert self.size() == self.episode_limit
        episode_data = dict(
            obs=self.obs_buf,
            state=self.state_buf,
            actions=self.action_buf,
            actions_onehot=self.action_onehot_buf,
            available_actions=self.available_action_buf,
            rewards=self.reward_buf,
            terminated=self.terminal_buf,
            filled=self.filled_buf,
        )
        return episode_data

    def size(self) -> int:
        """Get current size of replay memory.

        Returns:
        - int: Current size of the replay memory.
        """
        return self.curr_size

    def __len__(self) -> int:
        """Get the length of the EpisodeData.

        Returns:
        - int: Length of the EpisodeData.
        """
        return self.curr_size


class ReplayBuffer:
    """Multi-agent replay buffer for storing and sampling episode data."""

    def __init__(
        self,
        max_size: int,
        num_agents: int,
        num_actions: int,
        episode_limit: int,
        obs_shape: Union[int, Tuple],
        state_shape: Union[int, Tuple],
        dtype: torch.dtype = torch.float32,
        device: str = 'cpu',
    ) -> None:
        """Initialize MaReplayBuffer.

        Parameters:
        - max_size (int): Maximum number of episodes to store in the buffer.
        - num_agents (int): Number of agents.
        - num_actions (int): Number of possible actions.
        - episode_limit (int): Maximum number of steps in an episode.
        - obs_shape (Union[int, Tuple]): Shape of the observation.
        - state_shape (Union[int, Tuple]): Shape of the state.
        - dtype (torch.dtype): Data type for PyTorch tensors.
        - device (str): Device for PyTorch tensors ('cpu' or 'cuda').
        """
        self.obs_buf = np.zeros(
            (max_size, episode_limit, num_agents, obs_shape))
        self.state_buf = np.zeros((max_size, episode_limit, state_shape))
        self.action_buf = np.zeros((max_size, episode_limit, num_agents))
        self.action_onehot_buf = np.zeros(
            (max_size, episode_limit, num_agents, num_actions))
        self.available_action_buf = np.zeros(
            (max_size, episode_limit, num_agents, num_actions))
        self.reward_buf = np.zeros((max_size, episode_limit, 1))
        self.terminal_buf = np.zeros((max_size, episode_limit, 1))
        self.filled_buf = np.zeros((max_size, episode_limit, 1))

        self.state_shape = state_shape
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.num_agents = num_agents

        self.max_size = max_size
        self.episode_limit = episode_limit
        self.dtype = dtype
        self.device = device

        # memory management
        self.curr_ptr = 0
        self.curr_size = 0

    def store(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        actions_onehot: np.ndarray,
        available_actions: np.ndarray,
        rewards: np.ndarray,
        terminated: np.ndarray,
        filled: np.ndarray,
    ) -> None:
        """Store a step of data in the replay buffer.

        Parameters:
        - obs (np.ndarray): Observation data.
        - state (np.ndarray): State data.
        - actions (np.ndarray): Actions taken.
        - actions_onehot (np.ndarray): Actions in one-hot encoding.
        - available_actions (np.ndarray): Available actions for each agent.
        - rewards (np.ndarray): Reward received.
        - terminated (np.ndarray): Termination flag.
        - filled (np.ndarray): Filled flag.
        """
        self.obs_buf[self.curr_ptr] = obs
        self.state_buf[self.curr_ptr] = state
        self.action_buf[self.curr_ptr] = actions
        self.action_onehot_buf[self.curr_ptr] = actions_onehot
        self.available_action_buf[self.curr_ptr] = available_actions

        self.reward_buf[self.curr_ptr] = rewards
        self.terminal_buf[self.curr_ptr] = terminated
        self.filled_buf[self.curr_ptr] = filled

        self.curr_ptr = (self.curr_ptr + 1) % self.max_size
        self.curr_size = min(self.curr_size + 1, self.max_size)

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """Convert a numpy array to a PyTorch tensor.

        Parameters:
        - array (np.ndarray): Numpy array to be converted.
        - copy (bool): Whether to copy the data.

        Returns:
        - torch.Tensor: PyTorch tensor.
        """
        if copy:
            return torch.tensor(array, dtype=self.dtype, device=self.device)
        return torch.as_tensor(array, dtype=self.dtype, device=self.device)

    def sample_batch(self,
                     batch_size: int,
                     to_torch: bool = True) -> Dict[str, torch.Tensor]:
        """Sample a batch from the replay buffer.

        Parameters:
        - batch_size (int): Batch size.

        Returns:
        - Dict[str, torch.Tensor]: Batch of experience samples.
        """
        idx = np.random.randint(self.curr_size, size=batch_size)

        batch = dict(
            obs_batch=self.obs_buf[idx],
            state_batch=self.state_buf[idx],
            actions_batch=self.action_buf[idx],
            actions_onehot_batch=self.action_onehot_buf[idx],
            available_actions_batch=self.available_action_buf[idx],
            reward_batch=self.reward_buf[idx],
            terminated_batch=self.terminal_buf[idx],
            filled_batch=self.filled_buf[idx],
        )

        if to_torch:
            for key, val in batch.items():
                batch[key] = self.to_torch(val)
        return batch

    def size(self) -> int:
        """Get current size of replay memory.

        Returns:
        - int: Current size of the replay memory.
        """
        return self.curr_size

    def __len__(self) -> int:
        """Get the length of the MaReplayBuffer.

        Returns:
        - int: Length of the MaReplayBuffer.
        """
        return self.curr_size
