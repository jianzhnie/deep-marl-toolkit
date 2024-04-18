import argparse
from typing import Tuple, Union

import numpy as np
import torch

from .base_buffer import BaseBuffer


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


class SharedReplayBuffer(BaseBuffer):
    """Buffer to store training data.

    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    :param obs_shape: (gym.Space) observation space of agents.
    :param state_shape: (gym.Space) centralized observation space of agents.
    :param action_shape: (gym.Space) action space for agents.
    """

    def __init__(
        self,
        num_envs: int,
        num_agents: int,
        episode_limit: int,
        obs_shape: Union[int, Tuple],
        state_shape: Union[int, Tuple],
        action_shape: Union[int, Tuple],
        reward_shape: Union[int, Tuple],
        done_shape: Union[int, Tuple],
        args: argparse.Namespace,
    ) -> None:
        super(SharedReplayBuffer, self).__init__(
            num_envs,
            num_agents,
            episode_limit,
            obs_shape,
            state_shape,
            action_shape,
            reward_shape,
            done_shape,
            args,
        )
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.episode_limit = episode_limit

        self.hidden_size = args.hidden_size
        self.rnn_layers = args.rnn_layers
        self.gamma = args.gamma
        self.use_gae = args.use_gae
        self.gae_lambda = args.gae_lambda
        self.use_popart = args.use_popart
        self.use_valuenorm = args.use_valuenorm
        self.use_proper_time_limits = args.use_proper_time_limits
        self.algorithm_name = args.algorithm_name

        self.obs = np.zeros(
            (self.episode_limit + 1, self.num_envs, *obs_shape),
            dtype=np.float32,
        )
        self.state = np.zeros(
            (self.episode_limit + 1, self.num_envs, *state_shape),
            dtype=np.float32,
        )
        self.actions = np.zeros(
            (self.episode_limit, self.num_envs, num_agents),
            dtype=np.float32,
        )
        self.rewards = np.zeros(
            (self.episode_limit, self.num_envs, 1),
            dtype=np.float32,
        )
        self.value_preds = np.zeros(
            (self.episode_limit + 1, self.num_envs, 1),
            dtype=np.float32,
        )
        self.returns = np.zeros_like(self.value_preds)
        self.advantages = np.zeros(
            (self.episode_limit, self.num_envs, 1),
            dtype=np.float32,
        )
        self.action_log_probs = np.zeros(
            (self.episode_limit, self.num_envs, action_shape),
            dtype=np.float32,
        )
        self.masks = np.ones(
            (self.episode_limit + 1, self.num_envs, num_agents, 1),
            dtype=np.float32,
        )
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)
        self.available_actions = np.zeros(
            (self.episode_limit + 1, self.num_envs, num_agents, *action_shape),
            dtype=np.float32,
        )
        self.rnn_hidden_states_actor = np.zeros(
            (
                self.episode_limit + 1,
                self.num_envs,
                self.num_agents,
                self.rnn_layers,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        self.rnn_hidden_states_critic = np.zeros_like(
            self.rnn_hidden_states_actor)

        self.curr_ptr = 0
        self.curr_size = 0

    def insert(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        masks: np.ndarray,
        bad_masks: np.ndarray,
        active_masks: np.ndarray,
        value_preds: np.ndarray,
        action_log_probs: np.ndarray,
        available_actions: np.ndarray,
        rnn_hidden_states_actor: np.ndarray,
        rnn_hidden_states_critic: np.ndarray,
    ) -> None:
        """Inserts data into the buffer.

        :param obs: (np.ndarray) observation data.
        :param state: (np.ndarray) centralized observation data.
        :param actions: (np.ndarray) action data.
        :param rewards: (np.ndarray) reward data.
        :param masks: (np.ndarray) mask data.
        :param bad_masks: (np.ndarray) bad mask data.
        :param active_masks: (np.ndarray) active mask data.
        :param value_preds: (np.ndarray) value prediction data.
        :param action_log_probs: (np.ndarray) action log probabilities.
        :param available_actions: (np.ndarray) available action data.
        :param rnn_hidden_states_actor: (np.ndarray) rnn hidden states for actor.
        :param rnn_hidden_states_critic: (np.ndarray) rnn hidden states for critic.
        """
        self.obs[self.curr_ptr + 1] = obs
        self.state[self.curr_ptr + 1] = state
        self.actions[self.curr_ptr] = actions
        self.rewards[self.curr_ptr] = rewards
        self.masks[self.curr_ptr + 1] = masks
        self.bad_masks[self.curr_ptr] = bad_masks
        self.active_masks[self.curr_ptr] = active_masks
        self.value_preds[self.curr_ptr] = value_preds
        self.action_log_probs[self.curr_ptr] = action_log_probs
        self.rnn_hidden_states_actor[self.curr_ptr +
                                     1] = rnn_hidden_states_actor
        self.rnn_hidden_states_critic[self.curr_ptr +
                                      1] = rnn_hidden_states_critic

        if bad_masks is not None:
            self.bad_masks[self.curr_ptr] = bad_masks
        if active_masks is not None:
            self.active_masks[self.curr_ptr] = active_masks
        if available_actions is not None:
            self.available_actions[self.curr_ptr] = available_actions

        self.curr_ptr = (self.curr_ptr + 1) % self.episode_limit

    def after_update(self) -> None:
        """Copy last timestep data to first index.

        Called after update to model.
        """
        self.state[0] = self.state[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.rnn_hidden_states_actor[0] = self.rnn_hidden_states_actor[
            -1].copy()
        self.rnn_hidden_states_critic[0] = self.rnn_hidden_states_critic[
            -1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def compute_returns(self, next_value: np.array, value_normalizer=None):
        """Compute returns either as discounted sum of rewards, or using GAE.

        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        """
        if self.use_proper_time_limits:
            if self.use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self.use_popart or self.use_valuenorm:
                        # step + 1
                        delta = (self.rewards[step] +
                                 self.gamma * value_normalizer.denormalize(
                                     self.value_preds[step + 1]) *
                                 self.masks[step + 1] -
                                 value_normalizer.denormalize(
                                     self.value_preds[step]))
                        gae = (delta + self.gamma * self.gae_lambda * gae *
                               self.masks[step + 1])
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[
                            step] = gae + value_normalizer.denormalize(
                                self.value_preds[step])
                    else:
                        delta = (self.rewards[step] +
                                 self.gamma * self.value_preds[step + 1] *
                                 self.masks[step + 1] - self.value_preds[step])
                        gae = (delta + self.gamma * self.gae_lambda *
                               self.masks[step + 1] * gae)
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self.use_popart or self.use_valuenorm:
                        self.returns[step] = (
                            self.returns[step + 1] * self.gamma *
                            self.masks[step + 1] + self.rewards[step]
                        ) * self.bad_masks[step + 1] + (1 - self.bad_masks[
                            step + 1]) * value_normalizer.denormalize(
                                self.value_preds[step])
                    else:
                        self.returns[step] = (
                            self.returns[step + 1] * self.gamma *
                            self.masks[step + 1] + self.rewards[step]
                        ) * self.bad_masks[step + 1] + (1 - self.bad_masks[
                            step + 1]) * self.value_preds[step]
        else:
            if self.use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self.use_popart or self.use_valuenorm:
                        if (self.algorithm_name == 'mat'
                                or self.algorithm_name == 'mat_dec'):
                            value_t = value_normalizer.denormalize(
                                self.value_preds[step])
                            value_t_next = value_normalizer.denormalize(
                                self.value_preds[step + 1])
                            rewards_t = self.rewards[step]

                            delta = (rewards_t + self.gamma *
                                     self.masks[step + 1] * value_t_next -
                                     value_t)
                            gae = (delta + self.gamma * self.gae_lambda *
                                   self.masks[step + 1] * gae)
                            self.advantages[step] = gae
                            self.returns[step] = gae + value_t
                        else:
                            delta = (self.rewards[step] +
                                     self.gamma * value_normalizer.denormalize(
                                         self.value_preds[step + 1]) *
                                     self.masks[step + 1] -
                                     value_normalizer.denormalize(
                                         self.value_preds[step]))
                            gae = (delta + self.gamma * self.gae_lambda *
                                   self.masks[step + 1] * gae)
                            self.returns[
                                step] = gae + value_normalizer.denormalize(
                                    self.value_preds[step])
                    else:
                        if self.algorithm_name == 'mat' or self.algo == 'mat_dec':
                            rewards_t = self.rewards[step]
                            mean_v_t = np.mean(self.value_preds[step],
                                               axis=-2,
                                               keepdims=True)
                            mean_v_t_next = np.mean(self.value_preds[step + 1],
                                                    axis=-2,
                                                    keepdims=True)
                            delta = (rewards_t + self.gamma *
                                     self.masks[step + 1] * mean_v_t_next -
                                     mean_v_t)

                            gae = (delta + self.gamma * self.gae_lambda *
                                   self.masks[step + 1] * gae)
                            self.advantages[step] = gae
                            self.returns[step] = gae + self.value_preds[step]

                        else:
                            delta = (self.rewards[step] +
                                     self.gamma * self.value_preds[step + 1] *
                                     self.masks[step + 1] -
                                     self.value_preds[step])
                            gae = (delta + self.gamma * self.gae_lambda *
                                   self.masks[step + 1] * gae)
                            self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = (self.returns[step + 1] * self.gamma *
                                          self.masks[step + 1] +
                                          self.rewards[step])

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        """Yield training data for MLP policies.

        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        episode_limit, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_limit * num_agents

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                'PPO requires the number of processes ({}) '
                '* number of steps ({}) * number of agents ({}) = {} '
                'to be greater than or equal to the number of PPO mini batches ({}).'
                ''.format(
                    n_rollout_threads,
                    episode_limit,
                    num_agents,
                    n_rollout_threads * episode_limit * num_agents,
                    num_mini_batch,
                ))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [
            rand[i * mini_batch_size:(i + 1) * mini_batch_size]
            for i in range(num_mini_batch)
        ]

        state = self.state[:-1].reshape(-1, *self.state.shape[3:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[3:])
        rnn_hidden_states_actor = self.rnn_hidden_states_actor[:-1].reshape(
            -1, *self.rnn_hidden_states_actor.shape[3:])
        rnn_hidden_states_critic = self.rnn_hidden_states_critic[:-1].reshape(
            -1, *self.rnn_hidden_states_critic.shape[3:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(
                -1, self.available_actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(
            -1, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N M Dim]-->[T N M Dim]-->[T*N*M,Dim]-->[index,Dim]
            state_batch = state[indices]
            obs_batch = obs[indices]
            rnn_hidden_states_actor_batch = rnn_hidden_states_actor[indices]
            rnn_hidden_states_critic_batch = rnn_hidden_states_critic[indices]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            yield (
                state_batch,
                obs_batch,
                rnn_hidden_states_actor_batch,
                rnn_hidden_states_critic_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                available_actions_batch,
            )

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        """Yield training data for non-chunked RNN training.

        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        """
        episode_limit, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * num_agents
        assert n_rollout_threads * num_agents >= num_mini_batch, (
            'PPO requires the number of processes ({})* number of agents ({}) '
            'to be greater than or equal to the number of '
            'PPO mini batches ({}).'.format(n_rollout_threads, num_agents,
                                            num_mini_batch))
        num_envs_per_batch = batch_size // num_mini_batch
        perm = torch.randperm(batch_size).numpy()

        state = self.state.reshape(-1, batch_size, *self.state.shape[3:])
        obs = self.obs.reshape(-1, batch_size, *self.obs.shape[3:])
        rnn_hidden_states_actor = self.rnn_hidden_states_actor.reshape(
            -1, batch_size, *self.rnn_hidden_states_actor.shape[3:])
        rnn_hidden_states_critic = self.rnn_hidden_states_critic.reshape(
            -1, batch_size, *self.rnn_hidden_states_critic.shape[3:])
        actions = self.actions.reshape(-1, batch_size, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions.reshape(
                -1, batch_size, self.available_actions.shape[-1])
        value_preds = self.value_preds.reshape(-1, batch_size, 1)
        returns = self.returns.reshape(-1, batch_size, 1)
        masks = self.masks.reshape(-1, batch_size, 1)
        active_masks = self.active_masks.reshape(-1, batch_size, 1)
        action_log_probs = self.action_log_probs.reshape(
            -1, batch_size, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, batch_size, 1)

        for start_ind in range(0, batch_size, num_envs_per_batch):
            state_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                state_batch.append(state[:-1, ind])
                obs_batch.append(obs[:-1, ind])
                rnn_states_batch.append(rnn_hidden_states_actor[0:1, ind])
                rnn_states_critic_batch.append(rnn_hidden_states_critic[0:1,
                                                                        ind])
                actions_batch.append(actions[:, ind])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[:-1, ind])
                value_preds_batch.append(value_preds[:-1, ind])
                return_batch.append(returns[:-1, ind])
                masks_batch.append(masks[:-1, ind])
                active_masks_batch.append(active_masks[:-1, ind])
                old_action_log_probs_batch.append(action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            # [N[T, dim]]
            T, N = self.episode_limit, num_envs_per_batch
            # These are all from_numpys of size (T, N, -1)
            state_batch = np.stack(state_batch, 1)
            obs_batch = np.stack(obs_batch, 1)
            actions_batch = np.stack(actions_batch, 1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, 1)
            value_preds_batch = np.stack(value_preds_batch, 1)
            return_batch = np.stack(return_batch, 1)
            masks_batch = np.stack(masks_batch, 1)
            active_masks_batch = np.stack(active_masks_batch, 1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch,
                                                  1)
            adv_targ = np.stack(adv_targ, 1)

            # States is just a (N, dim) from_numpy [N[1,dim]]
            rnn_states_batch = np.stack(rnn_states_batch).reshape(
                N, *self.rnn_hidden_states_actor.shape[3:])
            rnn_states_critic_batch = np.stack(
                rnn_states_critic_batch).reshape(
                    N, *self.rnn_hidden_states_critic.shape[3:])

            # Flatten the (T, N, ...) from_numpys to (T * N, ...)
            state_batch = _flatten(T, N, state_batch)
            obs_batch = _flatten(T, N, obs_batch)
            actions_batch = _flatten(T, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(T, N,
                                                   available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(T, N, value_preds_batch)
            return_batch = _flatten(T, N, return_batch)
            masks_batch = _flatten(T, N, masks_batch)
            active_masks_batch = _flatten(T, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(T, N,
                                                  old_action_log_probs_batch)
            adv_targ = _flatten(T, N, adv_targ)

            yield (
                state_batch,
                obs_batch,
                rnn_states_batch,
                rnn_states_critic_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                available_actions_batch,
            )

    def recurrent_generator(self, advantages, num_mini_batch,
                            data_chunk_length):
        """Yield training data for chunked RNN training.

        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param data_chunk_length: (int) length of sequence chunks with which to train RNN.
        """
        episode_limit, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_limit * num_agents
        data_chunks = batch_size // data_chunk_length  # [C=r*T*M/L]
        mini_batch_size = data_chunks // num_mini_batch

        rand = torch.randperm(data_chunks).numpy()
        sampler = [
            rand[i * mini_batch_size:(i + 1) * mini_batch_size]
            for i in range(num_mini_batch)
        ]

        if len(self.state.shape) > 4:
            state = (self.state[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(
                -1, *self.state.shape[3:]))
            obs = (self.obs[:-1].transpose(1, 2, 0, 3, 4,
                                           5).reshape(-1, *self.obs.shape[3:]))
        else:
            state = _cast(self.state[:-1])
            obs = _cast(self.obs[:-1])

        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])
        rnn_hidden_states_actor = (self.rnn_hidden_states_actor[:-1].transpose(
            1, 2, 0, 3, 4).reshape(-1,
                                   *self.rnn_hidden_states_actor.shape[3:]))
        rnn_hidden_states_critic = (
            self.rnn_hidden_states_critic[:-1].transpose(
                1, 2, 0, 3,
                4).reshape(-1, *self.rnn_hidden_states_critic.shape[3:]))

        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:-1])

        for indices in sampler:
            state_batch = []
            obs_batch = []
            rnn_hidden_states_actor_batch = []
            rnn_hidden_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for index in indices:
                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N M Dim]-->[N,M,T,Dim]-->[N*M*T,Dim]-->[L,Dim]
                state_batch.append(state[ind:ind + data_chunk_length])
                obs_batch.append(obs[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(
                        available_actions[ind:ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind +
                                                     data_chunk_length])
                return_batch.append(returns[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind +
                                                       data_chunk_length])
                old_action_log_probs_batch.append(
                    action_log_probs[ind:ind + data_chunk_length])
                adv_targ.append(advantages[ind:ind + data_chunk_length])
                # size [T+1 N M Dim]-->[T N M Dim]-->[N M T Dim]-->[N*M*T,Dim]-->[1,Dim]
                rnn_hidden_states_actor_batch.append(
                    rnn_hidden_states_actor[ind])
                rnn_hidden_states_critic_batch.append(
                    rnn_hidden_states_critic[ind])

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, Dim)
            state_batch = np.stack(state_batch, axis=1)
            obs_batch = np.stack(obs_batch, axis=1)

            actions_batch = np.stack(actions_batch, axis=1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch,
                                                   axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch,
                                                  axis=1)
            adv_targ = np.stack(adv_targ, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_hidden_states_actor_batch = np.stack(
                rnn_hidden_states_actor_batch).reshape(
                    N, *self.rnn_hidden_states_actor.shape[3:])
            rnn_hidden_states_critic_batch = np.stack(
                rnn_hidden_states_critic_batch).reshape(
                    N, *self.rnn_hidden_states_critic.shape[3:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            state_batch = _flatten(L, N, state_batch)
            obs_batch = _flatten(L, N, obs_batch)
            actions_batch = _flatten(L, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N,
                                                   available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N,
                                                  old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)

            yield (
                state_batch,
                obs_batch,
                rnn_hidden_states_actor_batch,
                rnn_hidden_states_critic_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                available_actions_batch,
            )
