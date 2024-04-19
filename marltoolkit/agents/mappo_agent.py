import argparse
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from marltoolkit.agents.base_agent import BaseAgent
from marltoolkit.modules.actors.r_actor import R_Actor
from marltoolkit.modules.critics.r_critic import R_Critic
from marltoolkit.modules.utils.valuenorm import ValueNorm
from marltoolkit.utils.model_utils import get_gard_norm, huber_loss, mse_loss


class MAPPOAgent(BaseAgent):
    """MAPPO Policy  class. Wraps actor and critic networks to compute actions
    and value function predictions.

    Args:
        args (argparse.Namespace): Arguments containing relevant model and policy information.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        # Initialize hyperparameters and other parameters
        self.device = args.device
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self.obs_space = args.obs_space
        self.share_obs_space = args.state_space
        self.action_space = args.action_space

        self.use_recurrent_policy = args.use_recurrent_policy
        self.use_max_grad_norm = args.use_max_grad_norm
        self.use_clipped_value_loss = args.use_clipped_value_loss
        self.use_huber_loss = args.use_huber_loss
        self.use_popart = args.use_popart
        self.use_valuenorm = args.use_valuenorm
        self.use_value_active_masks = args.use_value_active_masks
        self.use_policy_active_masks = args.use_policy_active_masks

        # Initialize actor and critic networks

        self.actor_model: R_Actor = R_Actor(args).to(self.device)
        self.critic_model: R_Critic = R_Critic(args).to(self.device)

        assert (
            (self.use_popart and self.use_valuenorm) is False
        ), 'self.use_popart and self.use_valuenorm can not be set True simultaneously'

        if self.use_popart:
            self.value_normalizer = self.critic_model.v_out
        elif self.use_valuenorm:
            self.value_normalizer = ValueNorm(1).to(self.device)
        else:
            self.value_normalizer = None

        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor_model.parameters(),
            lr=self.actor_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_model.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def get_actions(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        masks: torch.Tensor,
        available_actions: torch.Tensor = None,
        actor_rnn_states: torch.Tensor = None,
        critic_rnn_states: torch.Tensor = None,
        deterministic: bool = False,
    ):
        """Compute actions and value function predictions for the given inputs.

        Args:
            obs (torch.Tensor): Local agent inputs to the actor.
            state (torch.Tensor): Centralized input to the critic.
            masks (torch.Tensor): Denotes points at which RNN states should be reset.
            available_actions (torch.Tensor, optional): Denotes which actions are available to the agent.
            actor_rnn_states (torch.Tensor, optional): RNN states for the actor network.
            critic_rnn_states (torch.Tensor, optional): RNN states for the critic network.
            deterministic (bool, optional): Whether the action should be deterministic or sampled.


        Return:
            :return values: (torch.Tensor) value function predictions.
            :return actions: (torch.Tensor) actions to take.
            :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
            :return actor_rnn_states: (torch.Tensor) updated actor network RNN states.
            :return critic_rnn_states: (torch.Tensor) updated critic network RNN states.
        """
        actions, action_log_probs, actor_rnn_states = self.actor_model.forward(
            obs, masks, available_actions, actor_rnn_states, deterministic)

        values, critic_rnn_states = self.critic_model.forward(
            state, masks, critic_rnn_states)
        return values, actions, action_log_probs, actor_rnn_states, critic_rnn_states

    def get_values(
        self,
        state: torch.Tensor,
        masks: torch.Tensor,
        critic_rnn_states: torch.Tensor,
    ):
        """Get value function predictions.

        :param state (torch.Tensor): centralized input to the critic.
        :param masks: (torch.Tensor) denotes points at which RNN states should be reset.
        :param critic_rnn_states: (torch.Tensor) if critic is RNN, RNN states for critic.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic_model.forward(state, masks, critic_rnn_states)
        return values

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        masks: torch.Tensor,
        active_masks: torch.Tensor = None,
        available_actions: torch.Tensor = None,
        actor_rnn_states: torch.Tensor = None,
        critic_rnn_states: torch.Tensor = None,
    ):
        """Get action logprobs / entropy and value function predictions for
        actor update.

        :param obs (torch.Tensor): local agent inputs to the actor.
        :param state (torch.Tensor): centralized input to the critic.
        :param action: (torch.Tensor) actions whose log probabilites and entropy to compute.
        :param masks: (torch.Tensor) denotes points at which RNN states should be reset.
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)
        :param actor_rnn_states: (torch.Tensor) if actor is RNN, RNN states for actor.
        :param critic_rnn_states: (torch.Tensor) if critic is RNN, RNN states for critic.


        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.actor_model.evaluate_actions(
            obs, action, masks, active_masks, available_actions,
            actor_rnn_states)

        values, _ = self.critic_model.forward(state, masks, critic_rnn_states)
        return values, action_log_probs, dist_entropy

    def act(
        self,
        obs: torch.Tensor,
        masks: torch.Tensor,
        available_actions: torch.Tensor = None,
        actor_rnn_states: torch.Tensor = None,
        deterministic: torch.Tensor = False,
    ):
        """Compute actions using the given inputs.

        :param obs (torch.Tensor): local agent inputs to the actor.
        :param actor_rnn_states: (torch.Tensor) if actor is RNN, RNN states for actor.
        :param masks: (torch.Tensor) denotes points at which RNN states should be reset.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, actor_rnn_states = self.actor_model.forward(
            obs, masks, available_actions, actor_rnn_states, deterministic)
        return actions, actor_rnn_states

    def cal_value_loss(
        self,
        values: torch.Tensor,
        value_preds_batch: torch.Tensor,
        return_batch: torch.Tensor,
        active_masks_batch: torch.Tensor,
    ):
        """Calculate value function loss.

        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (
            values - value_preds_batch).clamp(-self.clip_param,
                                              self.clip_param)
        if self.use_popart or self.use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = (self.value_normalizer.normalize(return_batch) -
                             value_pred_clipped)
            error_original = self.value_normalizer.normalize(
                return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self.use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self.use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self.use_value_active_masks:
            value_loss = (value_loss *
                          active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(
        self,
        batch_data: Union[List[np.array], Tuple[np.array]],
        update_actor: bool = True,
    ):
        """Update actor and critic networks.

        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        (
            obs_batch,
            state_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
        ) = batch_data

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.evaluate_actions(
            obs_batch,
            state_batch,
            actions_batch,
            masks_batch,
            active_masks_batch,
            available_actions_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
        )
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = (torch.clamp(imp_weights, 1.0 - self.clip_param,
                             1.0 + self.clip_param) * adv_targ)

        if self.use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) *
                active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(
                torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor_model.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.actor_model.parameters())

        self.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch,
                                         return_batch, active_masks_batch)

        self.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self.use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.critic_model.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.critic_model.parameters())

        self.critic_optimizer.step()

        return (
            policy_loss,
            value_loss,
            dist_entropy,
            actor_grad_norm,
            critic_grad_norm,
            imp_weights,
        )

    def learn(self, buffer, update_actor: bool = True):
        """Perform a training update using minibatch GD.

        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self.use_popart or self.use_valuenorm:
            advantages = buffer.returns[:
                                        -1] - self.value_normalizer.denormalize(
                                            buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = buffer.recurrent_generator(
                    advantages, self.num_mini_batch, self.data_chunk_length)

            data_generator = buffer.feed_forward_generator(
                advantages, self.num_mini_batch)

            for batch_data in data_generator:
                (
                    policy_loss,
                    value_loss,
                    dist_entropy,
                    actor_grad_norm,
                    critic_grad_norm,
                    imp_weights,
                ) = self.ppo_update(batch_data, update_actor)

                train_info['policy_loss'] += policy_loss.item()
                train_info['value_loss'] += value_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        self.actor_model.train()
        self.critic_model.train()

    def prep_rollout(self):
        self.actor_model.eval()
        self.critic_model.eval()
