from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

import numpy as np
import torch


class BaseBuffer(ABC):
    """Abstract base class for all buffers."""

    def __init__(
        self,
        n_envs: int,
        buffer_size: int,
        num_agents: int,
        state_space: Union[int, Tuple],
        obs_space: Union[int, Tuple],
        action_space: Union[int, Tuple],
        reward_space: Union[int, Tuple],
        done_space: Union[int, Tuple],
        **kwargs,
    ):

        self.num_agents = num_agents
        self.state_space = state_space
        self.obs_space = obs_space
        self.action_space = action_space
        self.reward_space = reward_space
        self.done_space = done_space
        self.n_envs = n_envs
        self.buffer_size = buffer_size

    @abstractmethod
    def store(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def reset_buffer(self):
        raise NotImplementedError

    @abstractmethod
    def sample(self, **kwargs):
        raise NotImplementedError

    def store_transitions(self, **kwargs):
        raise NotImplementedError

    def store_episodes(self, **kwargs):
        return NotImplementedError

    def finish_path(self, **kwargs):
        return NotImplementedError


class EpisodeData(object):

    def __init__(self, episode_limit: int, state_space: Union[int, Tuple],
                 obs_space: Union[int,
                                  Tuple], num_actions: int, num_agents: int):

        self.obs_buf = np.zeros((episode_limit, num_agents, obs_space))
        self.state_buf = np.zeros((episode_limit, state_space))
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

    def add(self, state, obs, actions, actions_onehot, available_actions,
            rewards, terminated, filled):

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

    def fill_mask(self):
        assert self.size() < self.episode_limit
        self.terminal_buf[self._curr_ptr] = True
        self.filled_buf[self._curr_ptr] = 1.0
        self._curr_ptr += 1
        self._curr_size = min(self._curr_size + 1, self.episode_limit)

    def get_data(self):
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


class EpisodeExperience(object):

    def __init__(self, episode_limit):
        self.episode_limit = episode_limit

        self.episode_state = []
        self.episode_actions = []
        self.episode_actions_onehot = []
        self.episode_reward = []
        self.episode_terminated = []
        self.episode_obs = []
        self.episode_available_actions = []
        self.episode_filled = []

    @property
    def count(self):
        return len(self.episode_state)

    def add(self, state, obs, actions, actions_onehot, available_actions,
            reward, terminated, filled):
        assert self.count < self.episode_limit
        self.episode_state.append(state)
        self.episode_obs.append(obs)
        self.episode_actions.append(actions)
        self.episode_actions_onehot.append(actions_onehot)
        self.episode_available_actions.append(available_actions)
        self.episode_reward.append(reward)
        self.episode_terminated.append(terminated)
        self.episode_filled.append(filled)

    def get_data(self):
        """Get all the data in an episode."""
        assert self.count == self.episode_limit
        episode_data = dict(
            state=np.array(self.episode_state),
            obs=np.array(self.episode_obs),
            actions=np.array(self.episode_actions),
            rewards=np.array(self.episode_reward),
            terminated=np.array(self.episode_terminated),
            available_actions=np.array(self.episode_available_actions),
            filled=np.array(self.episode_filled))
        return episode_data


class MaReplayBuffer(object):

    def __init__(self,
                 buffer_size: int,
                 episode_limit: int,
                 state_space: Union[int, Tuple],
                 obs_space: Union[int, Tuple],
                 num_agents: int,
                 num_actions: int,
                 dtype: torch.dtype = torch.float32,
                 device: str = 'cpu'):

        self.obs_buf = np.zeros(
            (buffer_size, episode_limit, num_agents, obs_space))
        self.state_buf = np.zeros((buffer_size, episode_limit, state_space))
        self.action_buf = np.zeros((buffer_size, episode_limit, num_agents))
        self.action_onehot_buf = np.zeros(
            (buffer_size, episode_limit, num_agents, num_actions))
        self.available_action_buf = np.zeros(
            (buffer_size, episode_limit, num_agents, num_actions))
        self.reward_buf = np.zeros((buffer_size, episode_limit, 1))
        self.terminal_buf = np.zeros((buffer_size, episode_limit, 1))
        self.filled_buf = np.zeros((buffer_size, episode_limit, 1))

        self.state_space = state_space
        self.obs_space = obs_space
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


class OffPolicyBuffer(BaseBuffer):
    """Replay buffer for off-policy MARL algorithms.

    Args:
        n_agents: number of agents.
        state_space: global  state shape, type: Discrete, Box.
        obs_space: observation space for one agent (suppose same obs space for group agents).
        action_space: action space for one agent (suppose same actions space for group agents).
        reward_space: reward space.
        done_space: terminal variable space.
        n_envs: number of parallel environments.
        buffer_size: buffer size for one environment.
        **kwargs: other arguments.
    """

    def __init__(
        self,
        num_envs: int,
        buffer_size: int,
        num_agents: int,
        state_space: Union[int, Tuple],
        obs_space: Union[int, Tuple],
        action_space: Union[int, Tuple],
        reward_space: Union[int, Tuple],
        done_space: Union[int, Tuple],
        **kwargs,
    ):
        super(OffPolicyBuffer, self).__init__(
            num_envs,
            buffer_size,
            num_agents,
            state_space,
            obs_space,
            action_space,
            reward_space,
            done_space,
        )

        self.max_size = buffer_size // num_envs
        if self.state_space is not None:
            self.store_global_state = True
        else:
            self.store_global_state = False
        self.buffers = {}
        self.reset_buffer()
        self.buffer_keys = self.buffers.keys()
        # memory management
        self._curr_ptr = 0
        self._curr_size = 0

    def reset_buffer(self):
        if self.store_global_state:
            self.state_buf = np.zeros((self.n_envs, self.max_size) +
                                      self.state_space)
            self.state_next = np.zeros((self.n_envs, self.max_size) +
                                       self.state_space)
        self.obs_buf = np.zeros((self.n_envs, self.max_size) + self.obs_space)
        self.obs_next_buf = np.zeros((self.n_envs, self.max_size) +
                                     self.obs_space)
        self.action_buf = np.zeros((self.n_envs, self.max_size) +
                                   self.action_space)
        self.action_onehot_buf = np.zeros((self.n_envs, self.max_size,
                                           self.num_agents) +
                                          self.action_space)
        self.available_action_buf = np.zeros((self.n_envs, self.max_size,
                                              self.num_agents) +
                                             self.action_space)
        self.agent_mask_buf = np.zeros(
            (self.n_envs, self.max_size, self.num_agents)).astype(np.bool)
        self.reward_buf = np.zeros((self.n_envs, self.max_size) +
                                   self.reward_space)
        self.terminal_buf = np.zeros((self.n_envs, self.max_size) +
                                     self.done_space).astype(np.bool)
        self.filled_buf = np.zeros((self.n_envs, self.max_size) +
                                   self.done_space).astype(np.bool)

        self.buffers = dict(
            obs=np.zeros((self.n_envs, self.max_size) + self.obs_space),
            obs_next=np.zeros((self.n_envs, self.max_size) + self.obs_space),
            actions=np.zeros((self.n_envs, self.max_size) + self.action_space),
            actions_onehot=np.zeros((self.n_envs, self.max_size,
                                     self.num_agents) + self.action_space),
            available_actions=np.zeros((self.n_envs, self.max_size,
                                        self.num_agents) + self.action_space),
            agent_mask=np.zeros(
                (self.n_envs, self.max_size, self.num_agents)).astype(np.bool),
            rewards=np.zeros((self.n_envs, self.max_size) + self.reward_space),
            terminated=np.zeros((self.n_envs, self.max_size) +
                                self.done_space).astype(np.bool),
            filled=np.zeros((self.n_envs, self.max_size) +
                            self.done_space).astype(np.bool),
        )
        if self.store_global_state:
            self.buffers['state'] = np.zeros((self.n_envs, self.max_size) +
                                             self.state_space)
            self.buffers['state_next'] = np.zeros((self.n_envs,
                                                   self.max_size) +
                                                  self.state_space)

    def store(self, step_data: Dict[str, np.ndarray]) -> None:
        for k in self.buffer_keys:
            assert k in step_data.keys(), f'{k} not in step_data'
            self.buffers[k][self._curr_ptr] = step_data[k]
        self._curr_ptr = (self._curr_ptr + 1) % self.buffer_size
        self._curr_size = min(self._curr_size + 1, self.buffer_size)

    def sample_batch(self, batch_size: int = None) -> Dict[str, np.ndarray]:
        """sample a batch from replay memory.

        Args:
            batch_size (int): batch size

        Returns:
            a batch of experience samples: obs, action, reward, next_obs, terminal
        """
        assert batch_size < self._curr_size, f'Batch Size: {batch_size} is larger than the current Buffer Size:{self._curr_size}'
        env_idxs = np.random.randint(self.n_envs, size=batch_size)
        step_idxs = np.random.randint(self._curr_size, size=batch_size)

        batch = {
            key: self.buffers[key][env_idxs, step_idxs]
            for key in self.buffer_keys
        }
        return batch

    def size(self) -> int:
        """get current size of replay memory."""
        return self._curr_size

    def __len__(self):
        return self._curr_size


class OffPolicyBufferRNN(OffPolicyBuffer):
    """Replay buffer for off-policy MARL algorithms with DRQN trick.

    Args:
        n_agents: number of agents.
        state_space: global  state shape, type: Discrete, Box.
        obs_space: observation space for one agent (suppose same obs space for group agents).
        action_space: action space for one agent (suppose same actions space for group agents).
        reward_space: reward space.
        done_space: terminal variable space.
        n_envs: number of parallel environments.
        buffer_size: buffer size for one environment.
        **kwargs: other arguments.
    """

    def __init__(
        self,
        num_envs: int,
        buffer_size: int,
        num_agents: int,
        state_space: Union[int, Tuple],
        obs_space: Union[int, Tuple],
        action_space: Union[int, Tuple],
        reward_space: Union[int, Tuple],
        done_space: Union[int, Tuple],
        **kwargs,
    ):
        super(OffPolicyBufferRNN, self).__init__(
            num_envs,
            buffer_size,
            num_agents,
            state_space,
            obs_space,
            action_space,
            reward_space,
            done_space,
        )
        self.max_eps_len = kwargs['max_episode_len']

    def reset_buffer(self):
        self.buffers = dict(
            obs=np.zeros((self.n_envs, self.max_size, self.num_agents,
                          self.max_eps_len) + self.obs_space),
            obs_next=np.zeros((self.n_envs, self.num_agents,
                               self.max_eps_len) + self.obs_space),
            actions=np.zeros((self.n_envs, self.num_agents, self.max_eps_len) +
                             self.action_space),
            actions_onehot=np.zeros((self.n_envs, self.num_agents,
                                     self.max_eps_len, self.num_agents) +
                                    self.action_space),
            available_actions=np.zeros((self.n_envs, self.num_agents,
                                        self.max_eps_len, self.num_agents) +
                                       self.action_space),
            agent_mask=np.zeros(
                (self.n_envs, self.num_agents, self.max_eps_len,
                 self.num_agents)).astype(np.bool),
            rewards=np.zeros((self.n_envs, self.max_eps_len) +
                             self.reward_space),
            terminated=np.zeros((self.n_envs, self.max_eps_len) +
                                self.done_space).astype(np.bool),
            filled=np.zeros((self.n_envs, self.max_eps_len) +
                            self.done_space).astype(np.bool),
        )
        if self.store_global_state:
            self.buffers['state'] = np.zeros((self.n_envs, self.max_eps_len +
                                              1) + self.state_space)
            self.buffers['state_next'] = np.zeros((self.n_envs,
                                                   self.max_eps_len) +
                                                  self.state_space)

    def store(self, step_data: Dict[str, np.ndarray]) -> None:
        for k in self.buffer_keys:
            assert k in step_data.keys(), f'{k} not in step_data'
            self.buffers[k][self._curr_ptr] = step_data[k]
        self._curr_ptr = (self._curr_ptr + 1) % self.buffer_size
        self._curr_size = min(self._curr_size + 1, self.buffer_size)

    def sample_batch(self, batch_size: int = None) -> Dict[str, np.ndarray]:
        """sample a batch from replay memory.

        Args:
            batch_size (int): batch size

        Returns:
            a batch of experience samples: obs, action, reward, next_obs, terminal
        """
        assert batch_size < self._curr_size, f'Batch Size: {batch_size} is larger than the current Buffer Size:{self._curr_size}'
        env_idxs = np.random.randint(self.n_envs, size=batch_size)
        step_idxs = np.random.randint(self._curr_size, size=batch_size)

        batch = {
            key: self.buffers[key][env_idxs, step_idxs]
            for key in self.buffer_keys
        }
        return batch

    def size(self) -> int:
        """get current size of replay memory."""
        return self._curr_size

    def __len__(self):
        return self._curr_size


class IndependReplayBuffer(object):

    def __init__(
        self,
        obs_dim: Union[int, Tuple],
        num_agents: int,
        buffer_size: int,
    ):

        self.obs_buf = np.zeros((buffer_size, num_agents, obs_dim),
                                dtype=np.float32)
        self.next_obs_buf = np.zeros((buffer_size, num_agents, obs_dim),
                                     dtype=np.float32)
        self.action_buf = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.reward_buf = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.terminal_buf = np.zeros((buffer_size, num_agents),
                                     dtype=np.float32)

        self._curr_ptr = 0
        self._curr_size = 0
        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.buffer_size = buffer_size

    def store(self, obs_all: List, act_all: List, reward_all: List,
              next_obs_all: List, terminal_all: List):
        agent_idx = 0
        for transition in zip(obs_all, act_all, reward_all, next_obs_all,
                              terminal_all):
            obs, act, reward, next_obs, terminal = transition

            self.obs_buf[self._curr_ptr, agent_idx] = obs
            self.next_obs_buf[self._curr_ptr, agent_idx] = next_obs
            self.action_buf[self._curr_ptr, agent_idx] = act
            self.reward_buf[self._curr_ptr, agent_idx] = reward
            self.terminal_buf[self._curr_ptr, agent_idx] = terminal

            agent_idx += 1

        self._curr_ptr = (self._curr_ptr + 1) % self.buffer_size
        self._curr_size = min(self._curr_size + 1, self.buffer_size)

    def sample_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        idxs = np.random.randint(self._curr_size, size=batch_size)

        batch = dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            action=self.action_buf[idxs],
            reward=self.reward_buf[idxs],
            terminal=self.terminal_buf[idxs],
            indices=idxs,  # for N -step Learning
        )

        return batch

    def sample_chunk(self, chunk_size: int,
                     batch_size: int) -> Dict[str, np.ndarray]:

        start_idx = np.random.randint(
            self._curr_size - chunk_size, size=batch_size)

        obs_chunk, next_obs_chunk, action_chunk, reward_chunk, terminal_chunk = [], [], [], [], []

        for idx in start_idx:
            obs = self.obs_buf[idx:idx + chunk_size]
            next_obs = self.next_obs_buf[idx:idx + chunk_size]
            action = self.action_buf[idx:idx + chunk_size]
            reward = self.reward_buf[idx:idx + chunk_size]
            terminal = self.terminal_buf[idx:idx + chunk_size]

            obs_chunk.append(obs)
            next_obs_chunk.append(next_obs)
            action_chunk.append(action)
            reward_chunk.append(reward)
            terminal_chunk.append(terminal)

        obs_chunk = np.stack(obs_chunk, axis=0)
        next_obs_chunk = np.stack(next_obs_chunk, axis=0)
        action_chunk = np.stack(action_chunk, axis=0)
        reward_chunk = np.stack(reward_chunk, axis=0)
        terminal_chunk = np.stack(terminal_chunk, axis=0)

        batch = dict(
            obs=obs_chunk,
            next_obs=next_obs_chunk,
            action=action_chunk,
            reward=reward_chunk,
            terminal=terminal_chunk)

        return batch

    def size(self) -> int:
        """get current size of replay memory."""
        return self._curr_size

    def __len__(self):
        return self._curr_size


if __name__ == '__main__':

    x = [1, 2]
    y = [2, 3]
    z = [False, True]

    for _ in zip(x, y, z):
        print(_)
