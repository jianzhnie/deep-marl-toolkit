from typing import Dict, Tuple, Union

import numpy as np

from marltoolkit.data.base_buffer import BaseBuffer


class OnPolicyBuffer(BaseBuffer):
    """Replay buffer for on-policy MARL algorithms.

    Args:
        max_size: buffer size of transition data for one environment.
        num_envs: number of parallel environments.
        num_agents: number of agents.
        obs_shape: observation space, type: Discrete, Box.
        state_shape: global state space, type: Discrete, Box.
        action_shape: action space, type: Discrete, MultiDiscrete, Box.
        reward_shape: reward space.
        done_shape: terminal variable space.
        gamma: discount factor.
        use_gae: whether to use GAE trick.
        gae_lambda: gae lambda.
        use_advantage_norm: whether to use Advantage normalization trick.
    """

    def __init__(
        self,
        max_size: int,
        num_envs: int,
        num_agents: int,
        obs_shape: Union[int, Tuple],
        state_shape: Union[int, Tuple],
        action_shape: Union[int, Tuple],
        reward_shape: Union[int, Tuple],
        done_shape: Union[int, Tuple],
        gamma: float = 0.99,
        use_gae: bool = False,
        gae_lambda: float = 0.8,
        use_advantage_norm: bool = False,
        device: str = 'cpu',
        **kwargs,
    ) -> None:
        super(OnPolicyBuffer, self).__init__(
            num_envs,
            num_agents,
            obs_shape,
            state_shape,
            action_shape,
            reward_shape,
            done_shape,
            device,
        )
        # Adjust buffer size
        self.max_size = max_size // self.num_envs
        self.gamma = gamma
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.use_advantage_norm = use_advantage_norm
        self.buffers = {}
        self.start_ids = None
        self.reset()
        self.buffer_keys = self.buffers.keys()

        # memory management
        self.curr_ptr = 0
        self.curr_size = 0

    def reset(self):
        self.buffers = {
            'obs':
            np.zeros(
                (self.max_size, self.num_envs) + self.obs_shape,
                dtype=np.float32,
            ),
            'actions':
            np.zeros(
                (self.max_size, self.num_envs, self.num_agents),
                dtype=np.float32,
            ),
            'rewards':
            np.zeros(
                (self.max_size, self.num_envs) + self.reward_shape,
                dtype=np.float32,
            ),
            'returns':
            np.zeros(
                (self.max_size, self.num_envs) + self.reward_shape,
                dtype=np.float32,
            ),
            'values':
            np.zeros(
                (self.max_size, self.num_envs, self.num_agents, 1),
                dtype=np.float32,
            ),
            'log_pi_old':
            np.zeros(
                (self.max_size, self.num_envs, self.num_agents),
                dtype=np.float32,
            ),
            'advantages':
            np.zeros(
                (self.max_size, self.num_envs) + self.reward_shape,
                dtype=np.float32,
            ),
            'dones':
            np.zeros((self.max_size, self.num_envs) + self.done_shape,
                     dtype=np.bool_),
            'agent_mask':
            np.ones((self.max_size, self.num_envs, self.num_agents),
                    dtype=np.bool_),
        }
        if self.state_shape is not None:
            self.buffers.update({
                'state':
                np.zeros(
                    (self.max_size, self.num_envs) + self.state_shape,
                    dtype=np.float32,
                )
            })
        self.curr_ptr, self.curr_size = 0, 0
        self.start_ids = np.zeros(self.num_envs, np.int64)
        # the start index of the last episode for each env.

    def store(self, step_data: Dict[str, np.array]) -> None:
        """Store step data in the buffer.

        Args:
            step_data (Dict[str, np.array]): step data, including obs, actions, rewards, values, dones, etc.
        """
        step_data_keys = step_data.keys()
        for k in self.buffer_keys:
            if k == 'advantages':
                continue
            if k in step_data_keys:
                self.buffers[k][self.curr_ptr] = step_data[k]
        self.curr_ptr = (self.curr_ptr + 1) % self.max_size
        self.curr_size = min(self.curr_size + 1, self.max_size)

    def finish_path(
        self,
        value: np.array,
        env_idx: int,
        value_normalizer=None,
    ) -> None:
        # when an episode is finished
        if self.curr_size == 0:
            return
        if self.size() == self.max_size:
            path_slice = np.arange(self.start_ids[env_idx],
                                   self.max_size).astype(np.int32)
        else:
            path_slice = np.arange(self.start_ids[env_idx],
                                   self.curr_ptr).astype(np.int32)

        # calculate advantages and returns
        rewards = np.array(self.buffers['rewards'][path_slice, env_idx])
        vs = np.append(
            np.array(self.buffers['values'][path_slice, env_idx]),
            [value],
            axis=0,
        )
        dones = np.array(self.buffers['dones'][path_slice, env_idx])[:, :,
                                                                     None]
        returns = np.zeros_like(rewards)
        last_gae_lam = 0
        step_nums = len(path_slice)

        if self.use_gae:
            for t in reversed(range(step_nums)):
                delta = rewards[t] + (
                    1 - dones[t]) * self.gamma * vs[t + 1] - vs[t]
                last_gae_lam = (delta + (1 - dones[t]) * self.gamma *
                                self.gae_lambda * last_gae_lam)
                returns[t] = last_gae_lam + vs[t]
        else:
            returns = np.append(returns, [value], axis=0)
            for t in reversed(range(step_nums)):
                returns[t] = rewards[t] + (
                    1 - dones[t]) * self.gamma * returns[t + 1]

        advantages = returns - vs[:-1]
        self.buffers['returns'][path_slice, env_idx] = returns
        self.buffers['advantages'][path_slice, env_idx] = advantages
        self.start_ids[env_idx] = self.curr_ptr

    def sample(self, idx: int):
        assert (
            self.size() == self.max_size
        ), 'Not enough transitions for on-policy buffer to random sample'

        samples = {}
        env_choices, step_choices = divmod(self.max_size, idx)
        for k in self.buffer_keys:
            if k == 'advantages':
                adv_batch = self.buffers[k][step_choices, env_choices]
                if self.use_advantage_norm:
                    adv_batch = (adv_batch - np.mean(adv_batch)) / (
                        np.std(adv_batch) + 1e-8)
                samples[k] = adv_batch
            else:
                samples[k] = self.buffers[k][step_choices, env_choices]
        return samples


class OnPolicyBufferRNN(OnPolicyBuffer):

    def __init__(
        self,
        max_size: int,
        num_envs: int,
        num_agents: int,
        episode_limit: int,
        obs_shape: Union[int, Tuple],
        state_shape: Union[int, Tuple],
        action_shape: Union[int, Tuple],
        reward_shape: Union[int, Tuple],
        done_shape: Union[int, Tuple],
        gamma: float = 0.99,
        use_gae: bool = False,
        gae_lambda: float = 0.8,
        use_advantage_norm: bool = False,
        device: str = 'cpu',
        **kwargs,
    ) -> None:
        super(OnPolicyBufferRNN).__init__(
            max_size,
            num_envs,
            num_agents,
            obs_shape,
            state_shape,
            action_shape,
            reward_shape,
            done_shape,
            gamma,
            use_gae,
            gae_lambda,
            use_advantage_norm,
            device,
        )
        self.episode_limit = episode_limit
        self.reset_episode()
        self.episode_keys = self.episode_data.keys()

        # memory management
        self.curr_ptr = 0
        self.curr_size = 0

    def reset(self) -> None:
        self.buffers = {
            'obs':
            np.zeros(
                (self.max_size, self.episode_limit, self.num_envs) +
                self.obs_shape,
                np.float32,
            ),
            'actions':
            np.zeros(
                (self.max_size, self.episode_limit, self.num_envs,
                 self.num_agents),
                np.float32,
            ),
            'rewards':
            np.zeros(
                (self.max_size, self.episode_limit, self.num_envs) +
                self.reward_shape,
                np.float32,
            ),
            'returns':
            np.zeros(
                (self.max_size, self.episode_limit, self.num_envs) +
                self.reward_shape,
                np.float32,
            ),
            'values':
            np.zeros(
                (self.max_size, self.episode_limit, self.num_envs) +
                self.reward_shape,
                np.float32,
            ),
            'advantages':
            np.zeros(
                (self.max_size, self.episode_limit, self.num_envs) +
                self.reward_shape,
                np.float32,
            ),
            'log_pi_old':
            np.zeros(
                (self.max_size, self.episode_limit, self.num_envs) +
                self.reward_shape,
                np.float32,
            ),
            'dones':
            np.zeros(
                (self.max_size, self.episode_limit, self.num_envs) +
                self.done_shape,
                np.bool_,
            ),
            'avail_actions':
            np.ones(
                (self.max_size, self.episode_limit, self.num_envs) +
                self.action_shape,
                np.bool_,
            ),
            'filled':
            np.zeros(
                (self.max_size, self.episode_limit, self.num_envs) +
                self.done_shape,
                np.bool_,
            ),
        }
        if self.state_shape is not None:
            self.buffers.update({
                'state':
                np.zeros(
                    (self.max_size, self.episode_limit, self.num_envs) +
                    self.state_shape,
                    np.float32,
                )
            })

    def reset_episode(self) -> None:
        self.episode_data = {
            'obs':
            np.zeros(
                (self.episode_limit, self.num_envs) + self.obs_shape,
                dtype=np.float32,
            ),
            'actions':
            np.zeros(
                (self.episode_limit, self.num_envs, self.num_agents),
                dtype=np.float32,
            ),
            'rewards':
            np.zeros(
                (self.episode_limit, self.num_envs) + self.reward_shape,
                dtype=np.float32,
            ),
            'returns':
            np.zeros(
                (self.episode_limit, self.num_envs) + self.reward_shape,
                np.float32,
            ),
            'values':
            np.zeros(
                (self.episode_limit, self.num_envs, self.num_agents) +
                self.reward_shape,
                np.float32,
            ),
            'advantages':
            np.zeros(
                (self.episode_limit, self.num_envs, self.num_agents) +
                self.reward_shape,
                np.float32,
            ),
            'log_pi_old':
            np.zeros(
                (self.episode_limit, self.num_envs, self.num_agents),
                np.float32,
            ),
            'dones':
            np.zeros(
                (self.episode_limit, self.num_envs) + self.done_shape,
                dtype=np.bool_,
            ),
            'available_actions':
            np.ones(
                (self.episode_limit, self.num_envs) + self.action_shape,
                dtype=np.bool_,
            ),
            'filled':
            np.zeros((self.episode_limit, self.num_envs, 1), dtype=np.bool_),
        }
        if self.state_shape is not None:
            self.episode_data.update({
                'state':
                np.zeros(
                    (self.episode_limit, self.num_envs) + self.state_shape,
                    dtype=np.float32,
                ),
            })

    def store_transitions(self, transition_data: Tuple) -> None:
        obs, state, action, rewards, values, log_pi, terminated, avail_actions = (
            transition_data)
        self.episode_data['obs'] = obs
        self.episode_data['actions'] = action
        self.episode_data['rewards'] = rewards
        self.episode_data['values'] = values
        self.episode_data['log_pi_old'] = log_pi
        self.episode_data['terminals'] = terminated
        self.episode_data['avail_actions'] = avail_actions
        if self.state_shape is not None:
            self.episode_data['state'] = state

    def store_episodes(self) -> None:
        for k in self.buffer_keys:
            self.buffers[k][self.curr_ptr] = self.episode_data[k].copy()
        self.curr_ptr = (self.curr_ptr + 1) % self.max_size
        self.curr_size = min(self.curr_size + 1, self.max_size)
        self.reset_episode()

    def sample(self, indexes):
        assert (
            self.size() == self.max_size
        ), 'Not enough transitions for on-policy buffer to random sample'
        samples = {}
        filled_batch = self.buffers['filled'][indexes]
        samples['filled'] = filled_batch
        for k in self.buffer_keys:
            if k == 'filled':
                continue
            if k == 'advantages':
                adv_batch = self.buffers[k][indexes]
                if self.use_advantage_norm:
                    adv_batch_copy = adv_batch.copy()
                    filled_batch_n = filled_batch[:, None, :, :].repeat(
                        self.num_agents, axis=1)
                    adv_batch_copy[filled_batch_n == 0] = np.nan
                    adv_batch = (adv_batch - np.nanmean(adv_batch_copy)) / (
                        np.nanstd(adv_batch_copy) + 1e-8)
                samples[k] = adv_batch
            else:
                samples[k] = self.buffers[k][indexes]
        return samples
