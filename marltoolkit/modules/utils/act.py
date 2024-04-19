import argparse

import torch
import torch.nn as nn

from marltoolkit.modules.utils.distributions import (CustomBernoulli,
                                                     CustomCategorical,
                                                     DiagGaussian)


class ACTLayer(nn.Module):
    """MLP Module to compute actions.

    :param action_space: (gym.Space) action space.
    :param input_dim: (int) dimension of network input.
    :param use_orthogonal: (bool) whether to use orthogonal initialization.
    :param gain: (float) gain of the output layer of the network.
    """

    def __init__(self, args: argparse.Namespace):
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.multi_discrete = False
        self.mujoco_box = False
        action_space = args.action_space
        self.action_type = args.action_space.__class__.__name__

        if action_space.__class__.__name__ == 'Discrete':
            action_dim = action_space.n
            self.action_out = CustomCategorical(args.hidden_size, action_dim,
                                                args.use_orthogonal, args.gain)
        elif action_space.__class__.__name__ == 'Box':
            self.mujoco_box = True
            action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(args.input_dim, action_dim,
                                           args.use_orthogonal, args.gain)
        elif action_space.__class__.__name__ == 'MultiBinary':
            action_dim = action_space.shape[0]
            self.action_out = CustomBernoulli(args.input_dim, action_dim,
                                              args.use_orthogonal, args.gain)
        elif action_space.__class__.__name__ == 'MultiDiscrete':
            self.multi_discrete = True
            action_dims = action_space.high - action_space.low + 1
            self.action_outs = []
            for action_dim in action_dims:
                self.action_outs.append(
                    CustomCategorical(
                        args.input_dim,
                        action_dim,
                        args.use_orthogonal,
                        args.gain,
                    ))
            self.action_outs = nn.ModuleList(self.action_outs)
        else:  # discrete + continous
            self.mixed_action = True
            continous_dim = action_space[0].shape[0]
            discrete_dim = action_space[1].n
            self.action_outs = nn.ModuleList([
                DiagGaussian(
                    args.input_dim,
                    continous_dim,
                    args.use_orthogonal,
                    args.gain,
                ),
                CustomCategorical(
                    args.input_dim,
                    discrete_dim,
                    args.use_orthogonal,
                    args.gain,
                ),
            ])

    def forward(
        self,
        inputs: torch.Tensor,
        available_actions: torch.Tensor = None,
        deterministic: bool = False,
    ):
        """Compute actions and action logprobs from given input.

        :param inputs: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """
        if self.mixed_action:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(inputs)
                action = action_logit.mode(
                ) if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                actions.append(action.float())
                action_log_probs.append(action_log_prob)

            actions = torch.cat(actions, -1)
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1),
                                         -1,
                                         keepdim=True)

        elif self.multi_discrete:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(inputs)
                action = action_logit.mode(
                ) if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)

            actions = torch.cat(actions, -1)
            action_log_probs = torch.cat(action_log_probs, -1)

        elif self.mujoco_box:
            action_logits = self.action_out(inputs)
            actions = action_logits.mode(
            ) if deterministic else action_logits.sample()
            action_log_probs = action_logits.log_probs(actions)

        else:
            action_logits = self.action_out(inputs, available_actions)
            actions = action_logits.mode(
            ) if deterministic else action_logits.sample()
            action_log_probs = action_logits.log_probs(actions)

        return actions, action_log_probs

    def get_probs(self,
                  inputs: torch.Tensor,
                  available_actions: torch.Tensor = None):
        """Compute action probabilities from inputs.

        :param inputs: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)

        :return action_probs: (torch.Tensor)
        """
        if self.mixed_action or self.multi_discrete:
            action_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(inputs)
                action_prob = action_logit.probs
                action_probs.append(action_prob)
            action_probs = torch.cat(action_probs, -1)
        else:
            action_logits = self.action_out(inputs, available_actions)
            action_probs = action_logits.probs

        return action_probs

    def evaluate_actions(
        self,
        inputs: torch.Tensor,
        action: torch.Tensor,
        available_actions: torch.Tensor = None,
        active_masks: torch.Tensor = None,
    ):
        """Compute log probability and entropy of given actions.

        :param inputs: (torch.Tensor) input to network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        if self.mixed_action:
            a, b = action.split((2, 1), -1)
            b = b.long()
            action = [a, b]
            action_log_probs = []
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(inputs)
                action_log_probs.append(action_logit.log_probs(act))
                if active_masks is not None:
                    if len(action_logit.entropy().shape) == len(
                            active_masks.shape):
                        dist_entropy.append(
                            (action_logit.entropy() * active_masks).sum() /
                            active_masks.sum())
                    else:
                        dist_entropy.append((action_logit.entropy() *
                                             active_masks.squeeze(-1)).sum() /
                                            active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())

            action_log_probs = torch.sum(torch.cat(action_log_probs, -1),
                                         -1,
                                         keepdim=True)
            dist_entropy = dist_entropy[0] / 2.0 + dist_entropy[1] / 0.98

        elif self.multi_discrete:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(inputs)
                action_log_probs.append(action_logit.log_probs(act))
                if active_masks is not None:
                    dist_entropy.append(
                        (action_logit.entropy() *
                         active_masks.squeeze(-1)).sum() / active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())

            action_log_probs = torch.cat(action_log_probs, -1)
            dist_entropy = sum(dist_entropy) / len(dist_entropy)

        elif self.mujoco_box:
            action_logits = self.action_out(inputs)
            action_log_probs = action_logits.log_probs(action)
            if active_masks is not None:
                dist_entropy = (
                    action_logits.entropy() *
                    active_masks.squeeze(-1)).sum() / active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()

        else:
            action_logits = self.action_out(inputs, available_actions)
            action_log_probs = action_logits.log_probs(action)
            if active_masks is not None:
                dist_entropy = (
                    action_logits.entropy() *
                    active_masks.squeeze(-1)).sum() / active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()

        return action_log_probs, dist_entropy

    def evaluate_actions_trpo(
        self,
        inputs: torch.Tensor,
        action: torch.Tensor,
        available_actions: torch.Tensor = None,
        active_masks: torch.Tensor = None,
    ):
        """Compute log probability and entropy of given actions.

        :param inputs: (torch.Tensor) input to network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """

        if self.multi_discrete:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            mu_collector = []
            std_collector = []
            probs_collector = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(inputs)
                mu = action_logit.mean
                std = action_logit.stddev
                action_log_probs.append(action_logit.log_probs(act))
                mu_collector.append(mu)
                std_collector.append(std)
                probs_collector.append(action_logit.logits)
                if active_masks is not None:
                    dist_entropy.append(
                        (action_logit.entropy() *
                         active_masks.squeeze(-1)).sum() / active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())
            action_mu = torch.cat(mu_collector, -1)
            action_std = torch.cat(std_collector, -1)
            all_probs = torch.cat(probs_collector, -1)
            action_log_probs = torch.cat(action_log_probs, -1)
            dist_entropy = torch.tensor(dist_entropy).mean()

        else:
            action_logits = self.action_out(inputs, available_actions)
            action_mu = action_logits.mean
            action_std = action_logits.stddev
            action_log_probs = action_logits.log_probs(action)
            if self.action_type == 'Discrete':
                all_probs = action_logits.logits
            else:
                all_probs = None
            if active_masks is not None:
                if self.action_type == 'Discrete':
                    dist_entropy = (
                        action_logits.entropy() *
                        active_masks.squeeze(-1)).sum() / active_masks.sum()
                else:
                    dist_entropy = (action_logits.entropy() *
                                    active_masks).sum() / active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()

        return action_log_probs, dist_entropy, action_mu, action_std, all_probs
