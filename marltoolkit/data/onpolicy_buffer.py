from typing import Dict, Tuple, Union

import numpy as np

from marltoolkit.data.base_buffer import BaseBuffer


class OnPolicyBuffer(BaseBuffer):
    """Replay buffer for on-policy MARL algorithms.

    Args:
        num_agents: number of agents.
        obs_space: observation space, type: Discrete, Box.
        state_space: global state space, type: Discrete, Box.
        action_space: action space, type: Discrete, MultiDiscrete, Box.
        reward_space: reward space.
        done_space: terminal variable space.
        num_envs: number of parallel environments.
        max_size: buffer size of transition data for one environment.
        use_gae: whether to use GAE trick.
        use_advnorm: whether to use Advantage normalization trick.
        gamma: discount factor.
        gae_lam: gae lambda.
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
        gamma: float = 0.99,
        use_gae: bool = False,
        gae_lambda: float = 0.8,
        use_advantage_norm: bool = False,
        device: str = 'cpu',
        **kwargs,
    ) -> None:
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
                (self.max_size, self.num_envs) + self.obs_space,
                dtype=np.float32,
            ),
            'actions':
            np.zeros(
                (self.max_size, self.num_envs, self.num_agents),
                dtype=np.float32,
            ),
            'rewards':
            np.zeros(
                (self.max_size, self.num_envs) + self.reward_space,
                dtype=np.float32,
            ),
            'returns':
            np.zeros(
                (self.max_size, self.num_envs) + self.reward_space,
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
                (self.max_size, self.num_envs) + self.reward_space,
                dtype=np.float32,
            ),
            'dones':
            np.zeros((self.max_size, self.num_envs) + self.done_space,
                     dtype=np.bool_),
            'agent_mask':
            np.ones((self.max_size, self.num_envs, self.num_agents),
                    dtype=np.bool_),
        }
        if self.state_space is not None:
            self.buffers.update({
                'state':
                np.zeros(
                    (self.max_size, self.num_envs) + self.state_space,
                    dtype=np.float32,
                )
            })
        self.curr_ptr, self.curr_size = 0, 0
        self.start_ids = np.zeros(self.num_envs, np.int64)
        # the start index of the last episode for each env.

    def store(self, step_data: Dict[str, np.array]) -> None:
        step_data_keys = step_data.keys()
        for k in self.buffer_keys:
            if k == 'advantages':
                continue
            if k in step_data_keys:
                self.buffers[k][self.curr_ptr] = step_data[k]
        self.curr_ptr = (self.curr_ptr + 1) % self.max_size
        self.curr_size = min(self.curr_size + 1, self.max_size)

    def finish_path(self,
                    value,
                    env_idx,
                    value_normalizer=None):  # when an episode is finished
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
        use_value_norm = False if (value_normalizer is None) else True

        if self.use_gae:
            for t in reversed(range(step_nums)):
                if use_value_norm:
                    vs_t, vs_next = (
                        value_normalizer.denormalize(vs[t]),
                        value_normalizer.denormalize(vs[t + 1]),
                    )
                else:
                    vs_t, vs_next = vs[t], vs[t + 1]
                delta = rewards[t] + (1 -
                                      dones[t]) * self.gamma * vs_next - vs_t
                last_gae_lam = (delta + (1 - dones[t]) * self.gamma *
                                self.gae_lambda * last_gae_lam)
                returns[t] = last_gae_lam + vs_t
            advantages = (returns - value_normalizer.denormalize(vs[:-1])
                          if use_value_norm else returns - vs[:-1])
        else:
            returns = np.append(returns, [value], axis=0)
            for t in reversed(range(step_nums)):
                returns[t] = rewards[t] + (
                    1 - dones[t]) * self.gamma * returns[t + 1]
            advantages = (returns - value_normalizer.denormalize(vs)
                          if use_value_norm else returns - vs)
            advantages = advantages[:-1]

        self.buffers['returns'][path_slice, env_idx] = returns
        self.buffers['advantages'][path_slice, env_idx] = advantages
        self.start_ids[env_idx] = self.curr_ptr

    def sample(self, idx):
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
        obs_space: Union[int, Tuple],
        state_space: Union[int, Tuple],
        action_space: Union[int, Tuple],
        reward_space: Union[int, Tuple],
        done_space: Union[int, Tuple],
        gamma: float = 0.99,
        use_gae: bool = False,
        gae_lambda: float = 0.8,
        use_advantage_norm: bool = False,
        device: str = 'cpu',
        **kwargs,
    ) -> None:
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
        self.episode_limit = kwargs.get('episode_length', None)
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
                self.obs_space,
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
                self.reward_space,
                np.float32,
            ),
            'returns':
            np.zeros(
                (self.max_size, self.episode_limit, self.num_envs) +
                self.reward_space,
                np.float32,
            ),
            'values':
            np.zeros(
                (self.max_size, self.episode_limit, self.num_envs) +
                self.reward_space,
                np.float32,
            ),
            'advantages':
            np.zeros(
                (self.max_size, self.episode_limit, self.num_envs) +
                self.reward_space,
                np.float32,
            ),
            'log_pi_old':
            np.zeros(
                (self.max_size, self.episode_limit, self.num_envs) +
                self.reward_space,
                np.float32,
            ),
            'dones':
            np.zeros(
                (self.max_size, self.episode_limit, self.num_envs) +
                self.done_space,
                np.bool_,
            ),
            'avail_actions':
            np.ones(
                (self.max_size, self.episode_limit, self.num_envs) +
                self.action_space,
                np.bool_,
            ),
            'filled':
            np.zeros(
                (self.max_size, self.episode_limit, self.num_envs) +
                self.done_space,
                np.bool_,
            ),
        }
        if self.state_space is not None:
            self.buffers.update({
                'state':
                np.zeros(
                    (self.max_size, self.episode_limit, self.num_envs) +
                    self.state_space,
                    np.float32,
                )
            })

    def reset_episode(self) -> None:
        self.episode_data = {
            'obs':
            np.zeros(
                (self.episode_limit, self.num_envs) + self.obs_space,
                dtype=np.float32,
            ),
            'actions':
            np.zeros(
                (self.episode_limit, self.num_envs, self.num_agents),
                dtype=np.float32,
            ),
            'rewards':
            np.zeros(
                (self.episode_limit, self.num_envs) + self.reward_space,
                dtype=np.float32,
            ),
            'returns':
            np.zeros(
                (self.episode_limit, self.num_envs) + self.reward_space,
                np.float32,
            ),
            'values':
            np.zeros(
                (self.episode_limit, self.num_envs, self.num_agents) +
                self.reward_space,
                np.float32,
            ),
            'advantages':
            np.zeros(
                (self.episode_limit, self.num_envs, self.num_agents) +
                self.reward_space,
                np.float32,
            ),
            'log_pi_old':
            np.zeros(
                (self.episode_limit, self.num_envs, self.num_agents),
                np.float32,
            ),
            'dones':
            np.zeros(
                (self.episode_limit, self.num_envs) + self.done_space,
                dtype=np.bool_,
            ),
            'available_actions':
            np.ones(
                (self.episode_limit, self.num_envs) + self.action_space,
                dtype=np.bool_,
            ),
            'filled':
            np.zeros((self.episode_limit, self.num_envs, 1), dtype=np.bool_),
        }
        if self.state_space is not None:
            self.episode_data.update({
                'state':
                np.zeros(
                    (self.episode_limit, self.num_envs) + self.state_space,
                    dtype=np.float32,
                ),
            })

    def store_transitions(self, transition_data):
        obs_n, actions_dict, state, rewards, terminated, avail_actions = transition_data
        self.episode_data['obs'] = obs_n
        self.episode_data['actions'] = actions_dict['actions']
        self.episode_data['rewards'] = rewards
        self.episode_data['values'] = actions_dict['values']
        self.episode_data['log_pi_old'] = actions_dict['log_pi']
        self.episode_data['terminals'] = terminated
        self.episode_data['avail_actions'] = avail_actions
        if self.state_space is not None:
            self.episode_data['state'] = state

    def store_episodes(self) -> None:
        episode_data_keys = self.episode_data.keys()
        for env_idx in range(self.num_envs):
            for k in self.buffer_keys:
                if k in episode_data_keys:
                    self.buffers[k][
                        self.curr_ptr] = self.episode_data[k][env_idx].copy()
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
