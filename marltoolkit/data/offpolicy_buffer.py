import warnings
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union

import numpy as np
import torch

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


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
        obs_space: Union[int, Tuple],
        state_space: Union[int, Tuple],
        action_space: Union[int, Tuple],
        reward_space: Union[int, Tuple],
        done_space: Union[int, Tuple],
        **kwargs,
    ):
        self.num_envs = num_envs
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
        self.episode_buffer = {}
        self.reset()
        self.episode_keys = self.episode_buffer.keys()

    def reset(self):
        self.episode_buffer = dict(
            obs=np.zeros(
                (self.num_envs, self.num_agents, self.episode_limit) +
                self.obs_space,
                dtype=np.float32,
            ),
            actions=np.zeros(
                (self.num_envs, self.num_agents, self.episode_limit) +
                self.action_space,
                dtype=np.int8,
            ),
            rewards=np.zeros(
                (self.num_envs, self.episode_limit) + self.reward_space,
                dtype=np.float32,
            ),
            dones=np.zeros(
                (self.num_envs, self.episode_limit) + self.done_space,
                dtype=bool),
            filled=np.zeros(
                (self.num_envs, self.episode_limit) + self.done_space,
                dtype=bool),
            available_actions=np.zeros(
                (self.num_envs, self.num_agents, self.episode_limit) +
                self.action_space,
                dtype=np.int8,
            ),
        )
        if self.store_global_state:
            self.episode_buffer['state'] = np.zeros(
                (self.num_envs, self.num_agents, self.episode_limit) +
                self.state_space,
                dtype=np.float32,
            )

        # memory management
        self.curr_ptr = 0
        self.curr_size = 0

    def store_transitions(self, transitions: Dict[str, np.ndarray]):
        """Store transitions in the buffer."""
        self.episode_buffer['obs'][:, :, self.curr_ptr] = transitions['obs']
        self.episode_buffer['actions'][:, :,
                                       self.curr_ptr] = transitions['actions']
        self.episode_buffer['rewards'][:, :,
                                       self.curr_ptr] = transitions['rewards']
        self.episode_buffer['dones'][:,
                                     self.curr_ptr] = transitions['env_dones']
        self.episode_buffer['available_actions'][:, :,
                                                 self.curr_ptr] = transitions[
                                                     'available_actions']
        if self.store_global_state:
            self.episode_buffer['state'][:,
                                         self.curr_ptr] = transitions['state']

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
        max_size: buffer size for one environment.
        **kwargs: other arguments.
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
        super().__init__(
            max_size,
            num_envs,
            num_agents,
            obs_space,
            state_space,
            action_space,
            reward_space,
            done_space,
            device,
        )

        # Adjust buffer size
        self.max_size = max(max_size // num_envs, 1)
        if self.state_space is not None:
            self.store_global_state = True
        else:
            self.store_global_state = False

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.buffers = {}
        self.reset()
        self.buffer_keys = self.buffers.keys()
        # memory management
        self.curr_ptr = 0
        self.curr_size = 0

        if psutil is not None:
            total_memory_usage: float = 0
            for k in self.buffer_keys:
                total_memory_usage += self.buffers[k].nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    'This system does not have apparently enough memory to store the complete '
                    f'replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB'
                )

    def reset(self) -> None:
        self.buffers = dict(
            obs=np.zeros(
                (
                    self.num_envs,
                    self.max_size,
                    self.num_agents,
                ) + self.obs_space,
                dtype=np.float32,
            ),
            next_obs=np.zeros(
                (
                    self.num_envs,
                    self.max_size,
                    self.num_agents,
                ) + self.obs_space,
                dtype=np.float32,
            ),
            actions=np.zeros(
                (
                    self.num_envs,
                    self.max_size,
                    self.num_agents,
                ) + self.action_space,
                dtype=np.int8,
            ),
            agent_mask=np.zeros(
                (
                    self.num_envs,
                    self.max_size,
                    self.num_agents,
                ),
                dtype=bool,
            ),
            rewards=np.zeros(
                (self.num_envs, self.max_size) + self.reward_space,
                dtype=np.float32),
            dones=np.zeros((self.num_envs, self.max_size) + self.done_space,
                           dtype=np.bool_),
            filled=np.zeros((self.num_envs, self.max_size) + self.done_space,
                            dtype=np.bool_),
        )
        if self.store_global_state:
            self.buffers['state'] = np.zeros(
                (self.num_envs, self.max_size) + self.state_space,
                dtype=np.float32)

    def store(self, step_data: Dict[str, np.ndarray]) -> None:
        for k in self.buffer_keys:
            assert k in step_data.keys(), f'{k} not in step_data'
            self.buffers[k][:, self.curr_ptr] = step_data[k]
        self.curr_ptr = (self.curr_ptr + 1) % self.max_size
        self.curr_size = min(self.curr_size + 1, self.max_size)

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
        env_idx = np.random.randint(self.num_envs, size=batch_size)
        step_idx = np.random.randint(self.curr_size, size=batch_size)

        batch = {
            key: self.buffers[key][env_idx, step_idx]
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
        max_size: buffer size for one environment.
        **kwargs: other arguments.
    """

    def __init__(
        self,
        max_size: int,
        num_envs: int,
        num_agents: int,
        episode_limit: int,
        obs_space: Union[int, Tuple],
        state_space: Union[int, Tuple],
        action_space: Union[int, Tuple],
        reward_space: Union[int, Tuple],
        done_space: Union[int, Tuple],
        device: Union[torch.device, str] = 'cpu',
        **kwargs,
    ):
        self.episode_limit = episode_limit
        super(OffPolicyBufferRNN, self).__init__(
            max_size,
            num_envs,
            num_agents,
            obs_space,
            state_space,
            action_space,
            reward_space,
            done_space,
            device=device,
        )
        self.episode_data = MaEpisodeData(num_envs, num_agents, episode_limit,
                                          obs_space, state_space, action_space,
                                          reward_space, done_space)

    def reset(self):
        self.buffers = dict(
            obs=np.zeros(
                (self.max_size, self.num_agents, self.episode_limit) +
                self.obs_space,
                dtype=np.float32,
            ),
            actions=np.zeros(
                (self.max_size, self.num_agents, self.episode_limit) +
                self.action_space,
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
            available_actions=np.zeros(
                (self.max_size, self.num_agents, self.episode_limit) +
                self.action_space,
                dtype=np.int8,
            ),
        )
        if self.store_global_state:
            self.buffers['state'] = np.zeros(
                (self.max_size, self.episode_limit) + self.state_space,
                dtype=np.float32)

    def store_episodes(self):
        for env_idx in range(self.num_envs):
            for k in self.buffer_keys:
                self.buffers[k][
                    self.curr_ptr] = self.episode_data.episode_buffer[k][
                        env_idx].copy()
        self.curr_ptr = (self.curr_ptr + 1) % self.max_size
        self.curr_size = min(self.curr_size + 1, self.max_size)
        self.episode_data.reset()

    def store_transitions(self, transitions: Dict[str, np.ndarray]):
        self.episode_data.store_transitions(transitions)

    def finish_path(self, env_idx: int, epi_step: int, *terminal_data) -> None:
        """Finish a trajectory path in the buffer.

        Args:
            env_idx (int): Environment index.
            epi_step (int): Episode step.
            terminal_data (Tuple): Terminal data.
        """
        next_obs, next_state, available_actions, filled = terminal_data
        self.episode_data.episode_buffer['obs'][env_idx, :,
                                                epi_step] = next_obs[env_idx]
        self.episode_data.episode_buffer['state'][
            env_idx, epi_step] = next_state[env_idx]
        self.episode_data.episode_buffer['available_actions'][
            env_idx, :, epi_step] = (available_actions[env_idx])
        self.episode_data.episode_buffer['filled'][env_idx] = filled[env_idx]

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
        step_idx = np.random.randint(self.curr_size, size=batch_size)

        batch = {key: self.buffers[key][step_idx] for key in self.buffer_keys}
        if to_torch:
            batch = self.to_torch(batch)
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
