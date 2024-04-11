from typing import Dict, Tuple, Union

import numpy as np
import torch

__all__ = ['EpisodeData', 'ReplayBuffer']


class EpisodeData:
    """Class for managing and storing episode data."""

    def __init__(
        self,
        num_agents: int,
        episode_limit: int,
        obs_space: Union[int, Tuple],
        state_space: Union[int, Tuple],
        action_space: Union[int, Tuple],
        reward_space: Union[int, Tuple],
        done_space: Union[int, Tuple],
        **kwargs,
    ) -> None:
        """Initialize EpisodeData.

        Parameters:
        - num_agents (int): Number of agents.
        - num_actions (int): Number of possible actions.
        - episode_limit (int): Maximum number of steps in an episode.
        - obs_space (Union[int, Tuple]): Shape of the observation.
        - state_space (Union[int, Tuple]): Shape of the state.
        """
        self.num_agents = num_agents
        self.episode_limit = episode_limit
        self.obs_space = obs_space
        self.state_space = state_space
        self.action_space = action_space
        self.reward_space = reward_space
        self.done_space = done_space

        if self.state_space is not None:
            self.store_global_state = True
        else:
            self.store_global_state = False

        # memory management
        self.curr_ptr = 0
        self.curr_size = 0
        self.reset()
        self.episode_keys = self.episode_buffer.keys()

    def reset(self):
        self.episode_buffer = dict(
            obs=np.zeros(
                (
                    self.episode_limit,
                    self.num_agents,
                ) + self.obs_space,
                dtype=np.float32,
            ),
            actions=np.zeros(
                (self.episode_limit, ) + (self.num_agents, ),
                dtype=np.int8,
            ),
            last_actions=np.zeros(
                (
                    self.episode_limit,
                    self.num_agents,
                ) + self.action_space,
                dtype=np.int8,
            ),
            available_actions=np.zeros(
                (
                    self.episode_limit,
                    self.num_agents,
                ) + self.action_space,
                dtype=np.int8,
            ),
            rewards=np.zeros(
                (self.episode_limit, ) + self.reward_space,
                dtype=np.float32,
            ),
            dones=np.zeros(
                (self.episode_limit, ) + self.done_space,
                dtype=np.bool_,
            ),
            filled=np.zeros(
                (self.episode_limit, ) + self.done_space,
                dtype=np.bool_,
            ),
        )
        if self.store_global_state:
            self.episode_buffer['state'] = np.zeros(
                (self.episode_limit, ) + self.state_space,
                dtype=np.float32,
            )

        # memory management
        self.curr_ptr = 0
        self.curr_size = 0

    def store_transitions(self, transitions: Dict[str, np.ndarray]) -> None:
        """Add a step of data to the episode.

        Parameters:
        - transitions (Dict[str, np.ndarray]): Transition data dictionary with the following keys
            - obs (np.ndarray): Observation data.
            - state (np.ndarray): State data.
            - actions (np.ndarray): Actions taken.
            - last_actions (np.ndarray): Actions in one-hot encoding.
            - available_actions (np.ndarray): Available actions for each agent.
            - rewards (np.ndarray): Reward received.
            - dones (np.ndarray): Termination flag.
            - filled (np.ndarray): Filled flag.
        """
        assert self.size() < self.episode_limit
        for key in self.episode_keys:
            self.episode_buffer[key][self.curr_ptr] = transitions[key]

        self.curr_ptr += 1
        self.curr_size = min(self.curr_size + 1, self.episode_limit)

    def fill_mask(self) -> None:
        """Fill the mask for the current step."""
        assert self.size() < self.episode_limit
        self.episode_buffer['filled'][self.curr_ptr] = True
        self.episode_buffer['dones'][self.curr_ptr] = True

        self.curr_ptr += 1
        self.curr_size = min(self.curr_size + 1, self.episode_limit)

    def get_episodes_data(self) -> Dict[str, np.ndarray]:
        """Get all the data in an episode.

        Returns:
        - Dict[str, np.ndarray]: Episode data dictionary.
        """
        assert self.size() == self.episode_limit
        return self.episode_buffer

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
        episode_limit: int,
        obs_space: Union[int, Tuple],
        state_space: Union[int, Tuple],
        action_space: Union[int, Tuple],
        reward_space: Union[int, Tuple],
        done_space: Union[int, Tuple],
        device: Union[torch.device, str] = 'cpu',
    ) -> None:
        """Initialize MaReplayBuffer.

        Parameters:
        - max_size (int): Maximum number of episodes to store in the buffer.
        - num_agents (int): Number of agents.
        - num_actions (int): Number of possible actions.
        - episode_limit (int): Maximum number of steps in an episode.
        - obs_space (Union[int, Tuple]): Shape of the observation.
        - state_space (Union[int, Tuple]): Shape of the state.
        - device (str): Device for PyTorch tensors ('cpu' or 'cuda').
        """
        self.max_size = max_size
        self.num_agents = num_agents
        self.episode_limit = episode_limit
        self.obs_space = obs_space
        self.state_space = state_space
        self.action_space = action_space
        self.reward_space = reward_space
        self.done_space = done_space
        self.device = device

        if self.state_space is not None:
            self.store_global_state = True
        else:
            self.store_global_state = False

        self.reset()
        self.buffer_keys = self.buffers.keys()
        # memory management
        self.curr_ptr = 0
        self.curr_size = 0
        # episode data buffer
        self.episode_data = EpisodeData(
            num_agents,
            episode_limit,
            obs_space,
            state_space,
            action_space,
            reward_space,
            done_space,
        )

    def reset(self) -> None:
        self.buffers = dict(
            obs=np.zeros(
                (
                    self.max_size,
                    self.episode_limit,
                    self.num_agents,
                ) + self.obs_space,
                dtype=np.float32,
            ),
            actions=np.zeros(
                (
                    self.max_size,
                    self.episode_limit,
                ) + (self.num_agents, ),
                dtype=np.int8,
            ),
            last_actions=np.zeros(
                (
                    self.max_size,
                    self.episode_limit,
                    self.num_agents,
                ) + self.action_space,
                dtype=np.int8,
            ),
            available_actions=np.zeros(
                (
                    self.max_size,
                    self.episode_limit,
                    self.num_agents,
                ) + self.action_space,
                dtype=np.int8,
            ),
            rewards=np.zeros(
                (
                    self.max_size,
                    self.episode_limit,
                ) + self.reward_space,
                dtype=np.float32,
            ),
            dones=np.zeros(
                (
                    self.max_size,
                    self.episode_limit,
                ) + self.done_space,
                dtype=np.bool_,
            ),
            filled=np.zeros(
                (
                    self.max_size,
                    self.episode_limit,
                ) + self.done_space,
                dtype=np.bool_,
            ),
        )
        if self.store_global_state:
            self.buffers['state'] = np.zeros(
                (
                    self.max_size,
                    self.episode_limit,
                ) + self.state_space,
                dtype=np.float32,
            )

        # memory management
        self.curr_ptr = 0
        self.curr_size = 0

    def store_episodes(self,
                       episode_buffer: Dict[str, np.ndarray] = None) -> None:
        """Store a episode data in the replay buffer.

        Parameters:
        - episode_buffer (Dict[str, np.ndarray]): Episode data dictionary with the following keys:
            - obs (np.ndarray): Observation data.
            - state (np.ndarray): State data.
            - actions (np.ndarray): Actions taken.
            - last_actions (np.ndarray): Actions in one-hot encoding.
            - available_actions (np.ndarray): Available actions for each agent.
            - rewards (np.ndarray): Reward received.
            - dones (np.ndarray): Termination flag.
            - filled (np.ndarray): Filled flag.
        """
        assert self.size() < self.max_size
        if episode_buffer is None:
            episode_buffer = self.episode_data.episode_buffer
        for key in self.buffer_keys:
            self.buffers[key][self.curr_ptr] = episode_buffer[key].copy()

        self.episode_data.reset()
        self.curr_ptr = (self.curr_ptr + 1) % self.max_size
        self.curr_size = min(self.curr_size + 1, self.max_size)

    def store_transitions(self, transitions: Dict[str, np.ndarray]) -> None:
        """Store a transition in the episode data.

        Args:
        - transitions (Dict[str, np.ndarray]): Transition data dictionary with the following keys:
            - obs (np.ndarray): Observation data.
            - state (np.ndarray): State data.
            - actions (np.ndarray): Actions taken.
            - last_actions (np.ndarray): Actions in one-hot encoding.
            - available_actions (np.ndarray): Available actions for each agent.
            - rewards (np.ndarray): Reward received.
            - dones (np.ndarray): Termination flag.
            - filled (np.ndarray): Filled flag.
        """
        self.episode_data.store_transitions(transitions)

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """Convert a numpy array to a PyTorch tensor.

        Parameters:
        - array (np.ndarray): Numpy array to be converted.
        - copy (bool): Whether to copy the data.

        Returns:
        - torch.Tensor: PyTorch tensor.
        """
        if copy:
            return torch.tensor(array, device=self.device)
        return torch.as_tensor(array, device=self.device)

    def sample(self,
               batch_size: int,
               to_torch: bool = True) -> Dict[str, torch.Tensor]:
        """Sample a batch from the replay buffer.

        Parameters:
        - batch_size (int): Batch size.

        Returns:
        - Dict[str, torch.Tensor]: Batch of experience samples.
        """
        assert (
            batch_size < self.curr_size
        ), f'Batch Size: {batch_size} is larger than the current Buffer Size:{self.curr_size}'

        idx = np.random.randint(self.curr_size, size=batch_size)

        batch = {key: self.buffers[key][idx] for key in self.buffer_keys}

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
