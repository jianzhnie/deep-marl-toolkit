from typing import Dict, Tuple, Union

import numpy as np
import torch


class EpisodeData(object):

    def __init__(
        self,
        episode_limit: int,
        state_shape: Union[int, Tuple],
        obs_shape: Union[int, Tuple],
        num_actions: int,
        num_agents: int,
    ):

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
        self._curr_ptr = 0
        self._curr_size = 0
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
    ):
        assert self.size() < self.episode_limit
        self.obs_buf[self._curr_ptr] = obs
        self.state_buf[self._curr_ptr] = state
        self.action_buf[self._curr_ptr] = actions
        self.action_onehot_buf[self._curr_ptr] = actions_onehot

        self.available_action_buf[self._curr_ptr] = available_actions

        self.reward_buf[self._curr_ptr] = rewards
        self.terminal_buf[self._curr_ptr] = terminated
        self.filled_buf[self._curr_ptr] = filled
        self._curr_ptr += 1
        self._curr_size = min(self._curr_size + 1, self.episode_limit)

    def fill_mask(self) -> None:
        assert self.size() < self.episode_limit
        self.terminal_buf[self._curr_ptr] = True
        self.filled_buf[self._curr_ptr] = 1.0
        self._curr_ptr += 1
        self._curr_size = min(self._curr_size + 1, self.episode_limit)

    def get_data(self) -> Dict[str, np.ndarray]:
        """Get all the data in an episode."""
        assert self.size() == self.episode_limit
        episode_data = dict(
            state=self.state_buf,
            obs=self.obs_buf,
            actions=self.action_buf,
            actions_onehot=self.action_onehot_buf,
            rewards=self.reward_buf,
            terminated=self.terminal_buf,
            available_actions=self.available_action_buf,
            filled=self.filled_buf)
        return episode_data

    def size(self) -> int:
        """get current size of replay memory."""
        return self._curr_size

    def __len__(self):
        return self._curr_size


class MaReplayBuffer(object):

    def __init__(self,
                 buffer_size: int,
                 episode_limit: int,
                 state_shape: Union[int, Tuple],
                 obs_shape: Union[int, Tuple],
                 num_agents: int,
                 num_actions: int,
                 dtype: torch.dtype = torch.float32,
                 device: str = 'cpu'):

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
        self._curr_ptr = 0
        self._curr_size = 0

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
    ):

        self.obs_buf[self._curr_ptr] = obs
        self.state_buf[self._curr_ptr] = state
        self.action_buf[self._curr_ptr] = actions
        self.action_onehot_buf[self._curr_ptr] = actions_onehot
        self.available_action_buf[self._curr_ptr] = available_actions

        self.reward_buf[self._curr_ptr] = rewards
        self.terminal_buf[self._curr_ptr] = terminated
        self.filled_buf[self._curr_ptr] = filled

        self._curr_ptr = (self._curr_ptr + 1) % self.buffer_size
        self._curr_size = min(self._curr_size + 1, self.buffer_size)

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return torch.tensor(array).to(self.device)
        return torch.as_tensor(array).to(self.device)

    def sample_batch(self, batch_size) -> Dict[str, np.ndarray]:
        """sample a batch from replay memory.

        Args:
            batch_size (int): batch size

        Returns:
            a batch of experience samples: obs, action, reward, next_obs, terminal
        """
        idxs = np.random.randint(self._curr_size, size=batch_size)

        batch = dict(
            state_batch=self.state_buf[idxs],
            obs_batch=self.obs_buf[idxs],
            actions_batch=self.action_buf[idxs],
            actions_onehot_batch=self.action_onehot_buf[idxs],
            available_actions_batch=self.available_action_buf[idxs],
            reward_batch=self.reward_buf[idxs],
            terminated_batch=self.terminal_buf[idxs],
            filled_batch=self.filled_buf[idxs],
        )
        batch = {
            key: torch.tensor(val, dtype=self.dtype, device=self.device)
            for (key, val) in batch.items()
        }

        return batch

    def size(self) -> int:
        """get current size of replay memory."""
        return self._curr_size

    def __len__(self):
        return self._curr_size
