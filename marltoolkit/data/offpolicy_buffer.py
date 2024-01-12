from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

import numpy as np
import torch


class BaseBuffer(ABC):
    """Abstract base class for all buffers.

    This class provides a basic structure for reinforcement learning experience
    replay buffers.
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
        device: Union[torch.device, str] = 'cpu',
        **kwargs,
    ):
        """Initialize the base buffer.

        :param num_envs: Number of environments.
        :param buffer_size: Maximum capacity of the buffer.
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
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.state_space = state_space
        self.obs_space = obs_space
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
        return NotImplementedError

    @abstractmethod
    def finish_path(self, **kwargs):
        """Finish a trajectory path in the buffer."""
        return NotImplementedError

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


class MaEpisodeData(object):

    def __init__(
        self,
        num_envs: int,
        num_agents: int,
        episode_limit: int,
        state_space: Union[int, Tuple],
        obs_space: Union[int, Tuple],
        action_space: Union[int, Tuple],
        reward_space: Union[int, Tuple],
        done_space: Union[int, Tuple],
        **kwargs,
    ):
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.episode_limit = episode_limit
        self.state_space = state_space
        self.obs_space = obs_space
        self.action_space = action_space
        self.reward_space = reward_space
        self.done_space = done_space

        if self.state_space is not None:
            self.store_global_state = True
        else:
            self.store_global_state = False
        self.episode_buffer = {}
        self.reset()
        self.episode_keys = self.episode_buffer.keys()

    def reset(self):
        self.episode_buffer = dict(
            obs=np.zeros(
                (self.num_envs, self.num_agents, self.episode_limit) +
                self.obs_space,
                dtype=np.float32),
            obs_next=np.zeros(
                (self.num_envs, self.num_agents, self.episode_limit) +
                self.obs_space,
                dtype=np.float32),
            actions=np.zeros(
                (self.num_envs, self.num_agents, self.episode_limit) +
                self.action_space,
                dtype=np.int8),
            actions_onehot=np.zeros(
                (self.num_envs, self.num_agents, self.episode_limit) +
                self.action_space,
                dtype=np.int8),
            available_actions=np.zeros(
                (self.num_envs, self.num_agents, self.episode_limit) +
                self.action_space,
                dtype=np.int8),
            rewards=np.zeros(
                (self.num_envs, self.episode_limit) + self.reward_space,
                dtype=np.float32),
            terminated=np.zeros((self.num_envs, self.episode_limit) +
                                self.done_space).astype(np.bool),
            filled=np.zeros((self.num_envs, self.episode_limit) +
                            self.done_space).astype(np.bool),
        )
        if self.store_global_state:
            self.episode_buffer['state'] = np.zeros(
                (self.num_envs, self.num_agents, self.episode_limit) +
                self.state_space,
                dtype=np.float32)
            self.episode_buffer['state_next'] = np.zeros(
                (self.num_envs, self.num_agents, self.episode_limit) +
                self.state_space,
                dtype=np.float32)

        # memory management
        self.curr_ptr = 0
        self.curr_size = 0

    def store_transitions(self, transitions: Dict[str, np.ndarray]):
        """Store transitions in the buffer."""
        for k in self.episode_keys:
            assert k in transitions.keys(), f'{k} not in transitions'
            self.episode_buffer[k][:, :self.curr_ptr] = transitions[k]

    def size(self) -> int:
        """get current size of replay memory."""
        return self.curr_size

    def __len__(self):
        return self.curr_size


class OffPolicyBuffer(BaseBuffer):
    """Replay buffer for off-policy MARL algorithms.

    Args:
        n_agents: number of agents.
        state_space: global state shape, type: Discrete, Box.
        obs_space: observation space for one agent (suppose same obs space for group agents).
        action_space: action space for one agent (suppose same actions space for group agents).
        reward_space: reward space.
        done_space: terminal variable space.
        num_envs: number of parallel environments.
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
        device: Union[torch.device, str] = 'cpu',
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
            device,
        )

        # Adjust buffer size
        self.buffer_size = max(buffer_size // num_envs, 1)
        if self.state_space is not None:
            self.store_global_state = True
        else:
            self.store_global_state = False
        self.buffers = {}
        self.reset()
        self.buffer_keys = self.buffers.keys()
        # memory management
        self.curr_ptr = 0
        self.curr_size = 0

    def reset(self) -> None:
        self.buffers = dict(
            obs=np.zeros(
                (self.num_envs, self.buffer_size, self.num_agents) +
                self.obs_space,
                dtype=np.float32),
            obs_next=np.zeros(
                (self.num_envs, self.buffer_size, self.num_agents) +
                self.obs_space,
                dtype=np.float32),
            actions=np.zeros(
                (self.num_envs, self.buffer_size, self.num_agents) +
                self.action_space,
                dtype=np.int8),
            actions_onehot=np.zeros(
                (self.num_envs, self.buffer_size, self.num_agents) +
                self.action_space,
                dtype=np.int8),
            available_actions=np.zeros(
                (self.num_envs, self.buffer_size, self.num_agents) +
                self.action_space,
                dtype=np.int8),
            agent_mask=np.zeros((self.num_envs, self.buffer_size,
                                 self.num_agents)).astype(np.bool),
            rewards=np.zeros(
                (self.num_envs, self.buffer_size) + self.reward_space,
                dtype=np.float32),
            terminated=np.zeros((self.num_envs, self.buffer_size) +
                                self.done_space).astype(np.bool),
            filled=np.zeros((self.num_envs, self.buffer_size) +
                            self.done_space).astype(np.bool),
        )
        if self.store_global_state:
            self.buffers['state'] = np.zeros(
                (self.num_envs, self.buffer_size) + self.state_space,
                dtype=np.float32)
            self.buffers['state_next'] = np.zeros(
                (self.num_envs, self.buffer_size) + self.state_space,
                dtype=np.float32)

    def store(self, step_data: Dict[str, np.ndarray]) -> None:
        for k in self.buffer_keys:
            assert k in step_data.keys(), f'{k} not in step_data'
            self.buffers[k][self.curr_ptr] = step_data[k]
        self.curr_ptr = (self.curr_ptr + 1) % self.buffer_size
        self.curr_size = min(self.curr_size + 1, self.buffer_size)

    def sample_batch(
        self,
        batch_size: int = 32,
        to_torch: bool = True,
    ) -> Dict[str, np.ndarray]:
        """sample a batch from replay memory.

        Args:
            batch_size (int): batch size

        Returns:
            a batch of experience samples: obs, action, reward, next_obs, terminal
        """
        assert batch_size < self.curr_size, f'Batch Size: {batch_size} is larger than the current Buffer Size:{self.curr_size}'
        env_idxs = np.random.randint(self.num_envs, size=batch_size)
        step_idxs = np.random.randint(self.curr_size, size=batch_size)

        batch = {
            key: self.buffers[key][env_idxs, step_idxs]
            for key in self.buffer_keys
        }
        if to_torch:
            batch = self.to_torch(batch)
        return batch


class OffPolicyBufferRNN(OffPolicyBuffer):
    """Replay buffer for off-policy MARL algorithms with DRQN trick.

    Args:
        n_agents: number of agents.
        state_space: global  state shape, type: Discrete, Box.
        obs_space: observation space for one agent (suppose same obs space for group agents).
        action_space: action space for one agent (suppose same actions space for group agents).
        reward_space: reward space.
        done_space: terminal variable space.
        num_envs: number of parallel environments.
        buffer_size: buffer size for one environment.
        **kwargs: other arguments.
    """

    def __init__(
        self,
        num_envs: int,
        buffer_size: int,
        num_agents: int,
        episode_limit: int,
        state_space: Union[int, Tuple],
        obs_space: Union[int, Tuple],
        action_space: Union[int, Tuple],
        reward_space: Union[int, Tuple],
        done_space: Union[int, Tuple],
        **kwargs,
    ):
        super(OffPolicyBufferRNN,
              self).__init__(num_envs, buffer_size, num_agents, state_space,
                             obs_space, action_space, reward_space, done_space)
        self.episode_limit = episode_limit
        self.episode_data = MaEpisodeData(num_envs, num_agents, episode_limit,
                                          state_space, obs_space, action_space,
                                          reward_space, done_space)

    def reset(self):
        self.buffers = dict(
            obs=np.zeros(
                (self.buffer_size, self.num_agents, self.episode_limit) +
                self.obs_space,
                dtype=np.float32),
            obs_next=np.zeros(
                (self.num_agents, self.episode_limit) + self.obs_space,
                dtype=np.float32),
            actions=np.zeros(
                (self.num_agents, self.episode_limit) + self.action_space,
                dtype=np.int8),
            actions_onehot=np.zeros(
                (self.num_agents, self.episode_limit, self.num_agents) +
                self.action_space,
                dtype=np.int8),
            available_actions=np.zeros(
                (self.num_agents, self.episode_limit, self.num_agents) +
                self.action_space,
                dtype=np.int8),
            rewards=np.zeros(
                (self.episode_limit) + self.reward_space, dtype=np.float32),
            terminated=np.zeros(
                (self.episode_limit) + self.done_space, dtype=np.bool),
            filled=np.zeros(
                (self.episode_limit) + self.done_space, dtype=np.bool),
        )
        if self.store_global_state:
            self.buffers['state'] = np.zeros(
                (self.buffer_size, self.episode_limit) + self.state_space,
                dtype=np.float32)
            self.buffers['state_next'] = np.zeros(
                (self.buffer_size, self.episode_limit) + self.state_space,
                dtype=np.float32)

    def store(self, step_data: Dict[str, np.ndarray]) -> None:
        for k in self.buffer_keys:
            assert k in step_data.keys(), f'{k} not in step_data'
            self.buffers[k][self.curr_ptr] = step_data[k]
        self.curr_ptr = (self.curr_ptr + 1) % self.buffer_size
        self.curr_size = min(self.curr_size + 1, self.buffer_size)

    def store_episodes(self):
        for env_idx in range(self.num_envs):
            for k in self.buffer_keys:
                self.buffers[k][
                    self.curr_ptr] = self.episode_data.episode_buffer[k][
                        env_idx].copy()
        self.curr_ptr = (self.curr_ptr + 1) % self.buffer_size
        self.curr_size = min(self.curr_size + 1, self.buffer_size)
        self.episode_data.reset()

    def store_transitions(self, transitions: Dict[str, np.ndarray]):
        self.episode_data.store_transitions(transitions)

    def sample(self,
               batch_size: int = 32,
               to_torch: bool = True) -> Dict[str, np.ndarray]:
        """sample a batch from replay memory.

        Args:
            batch_size (int): batch size

        Returns:
            a batch of experience samples: obs, action, reward, next_obs, terminal
        """
        assert batch_size < self.curr_size, f'Batch Size: {batch_size} is larger than the current Buffer Size:{self.curr_size}'
        step_idxs = np.random.randint(self.curr_size, size=batch_size)

        batch = {key: self.buffers[key][step_idxs] for key in self.buffer_keys}
        return batch

    def size(self) -> int:
        """get current size of replay memory."""
        return self.curr_size

    def __len__(self):
        return self.curr_size


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

        self.curr_ptr = 0
        self.curr_size = 0
        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.buffer_size = buffer_size

    def store(self, obs_all: List, act_all: List, reward_all: List,
              next_obs_all: List, terminal_all: List):
        agent_idx = 0
        for transition in zip(obs_all, act_all, reward_all, next_obs_all,
                              terminal_all):
            obs, act, reward, next_obs, terminal = transition

            self.obs_buf[self.curr_ptr, agent_idx] = obs
            self.next_obs_buf[self.curr_ptr, agent_idx] = next_obs
            self.action_buf[self.curr_ptr, agent_idx] = act
            self.reward_buf[self.curr_ptr, agent_idx] = reward
            self.terminal_buf[self.curr_ptr, agent_idx] = terminal

            agent_idx += 1

        self.curr_ptr = (self.curr_ptr + 1) % self.buffer_size
        self.curr_size = min(self.curr_size + 1, self.buffer_size)

    def sample_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        idxs = np.random.randint(self.curr_size, size=batch_size)

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
            self.curr_size - chunk_size, size=batch_size)

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
        return self.curr_size

    def __len__(self):
        return self.curr_size


if __name__ == '__main__':

    x = [1, 2]
    y = [2, 3]
    z = [False, True]

    for _ in zip(x, y, z):
        print(_)
