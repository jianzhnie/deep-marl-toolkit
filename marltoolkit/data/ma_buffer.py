from typing import Dict, Tuple, Union

import numpy as np
import torch


class EpisodeData:
    """Class for managing and storing episode data."""

    def __init__(
        self,
        episode_limit: int,
        state_shape: Union[int, Tuple],
        obs_shape: Union[int, Tuple],
        num_actions: int,
        num_agents: int,
    ):
        """Initialize EpisodeData.

        Parameters:
        - episode_limit (int): Maximum number of steps in an episode.
        - state_shape (Union[int, Tuple]): Shape of the state.
        - obs_shape (Union[int, Tuple]): Shape of the observation.
        - num_actions (int): Number of possible actions.
        - num_agents (int): Number of agents.
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
        state: np.ndarray,
        obs: np.ndarray,
        actions: np.ndarray,
        actions_onehot: np.ndarray,
        available_actions: np.ndarray,
        rewards: np.ndarray,
        terminated: np.ndarray,
        filled: np.ndarray,
    ) -> None:
        """Add a step of data to the episode.

        Parameters:
        - state (np.ndarray): State data.
        - obs (np.ndarray): Observation data.
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
        episode_data = dict(state=self.state_buf,
                            obs=self.obs_buf,
                            actions=self.action_buf,
                            actions_onehot=self.action_onehot_buf,
                            rewards=self.reward_buf,
                            terminated=self.terminal_buf,
                            available_actions=self.available_action_buf,
                            filled=self.filled_buf)
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


class MaReplayBuffer:
    """Multi-agent replay buffer for storing and sampling episode data."""

    def __init__(self,
                 buffer_size: int,
                 episode_limit: int,
                 state_shape: Union[int, Tuple],
                 obs_shape: Union[int, Tuple],
                 num_agents: int,
                 num_actions: int,
                 dtype: torch.dtype = torch.float32,
                 device: str = 'cpu'):
        """Initialize MaReplayBuffer.

        Parameters:
        - buffer_size (int): Maximum number of episodes to store in the buffer.
        - episode_limit (int): Maximum number of steps in an episode.
        - state_shape (Union[int, Tuple]): Shape of the state.
        - obs_shape (Union[int, Tuple]): Shape of the observation.
        - num_agents (int): Number of agents.
        - num_actions (int): Number of possible actions.
        - dtype (torch.dtype): Data type for PyTorch tensors.
        - device (str): Device for PyTorch tensors ('cpu' or 'cuda').
        """
        self.obs_buf = np.zeros(
            (buffer_size, episode_limit, num_agents, obs_shape))
        self.state_buf = np.zeros((buffer_size, episode_limit, state_shape))
        self.action_buf = np.zeros((buffer_size, episode_limit, num_agents))
        self.action_onehot_buf = np.zeros(
            (buffer_size, episode_limit, num_agents, num_actions))
        self.available_action_buf = np.zeros(
            (buffer_size, episode_limit, num_agents, num_actions))
        self.reward_buf = np.zeros((buffer_size, episode_limit, 1))
        self.terminal_buf = np.zeros((buffer_size, episode_limit, 1))
        self.filled_buf = np.zeros((buffer_size, episode_limit, 1))

        self.state_shape = state_shape
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.num_agents = num_agents

        self.buffer_size = buffer_size
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

        self.curr_ptr = (self.curr_ptr + 1) % self.buffer_size
        self.curr_size = min(self.curr_size + 1, self.buffer_size)

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

    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch from the replay buffer.

        Parameters:
        - batch_size (int): Batch size.

        Returns:
        - Dict[str, torch.Tensor]: Batch of experience samples.
        """
        idxs = np.random.randint(self.curr_size, size=batch_size)

        batch = dict(
            state_batch=self.to_torch(self.state_buf[idxs]),
            obs_batch=self.to_torch(self.obs_buf[idxs]),
            actions_batch=self.to_torch(self.action_buf[idxs]),
            actions_onehot_batch=self.to_torch(self.action_onehot_buf[idxs]),
            available_actions_batch=self.to_torch(
                self.available_action_buf[idxs]),
            reward_batch=self.to_torch(self.reward_buf[idxs]),
            terminated_batch=self.to_torch(self.terminal_buf[idxs]),
            filled_batch=self.to_torch(self.filled_buf[idxs]),
        )

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
