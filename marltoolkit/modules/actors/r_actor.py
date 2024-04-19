import torch
import torch.nn as nn

from marltoolkit.modules.utils.act import ACTLayer
from marltoolkit.modules.utils.common import MLPBase, RNNLayer


class R_Actor(nn.Module):
    """Actor network class for MAPPO. Outputs actions given observations.

    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args) -> None:
        super(R_Actor, self).__init__()

        self.use_recurrent_policy = args.use_recurrent_policy
        self.use_policy_active_masks = args.use_policy_active_masks
        self.base = MLPBase(
            input_dim=args.actor_input_dim,
            hidden_dim=args.hidden_size,
            activation=args.activation,
            use_orthogonal=args.use_orthogonal,
            use_feature_normalization=args.use_feature_normalization,
        )
        if args.use_recurrent_policy:
            self.rnn = RNNLayer(
                args.hidden_size,
                args.hidden_size,
                args.rnn_layers,
                args.use_orthogonal,
            )
        self.act = ACTLayer(args)
        self.algorithm_name = args.algorithm_name

    def forward(
        self,
        obs: torch.Tensor,
        masks: torch.Tensor,
        available_actions: torch.Tensor = None,
        rnn_hidden_states: torch.Tensor = None,
        deterministic: bool = False,
    ):
        """Compute actions from the given inputs.

        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param rnn_hidden_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.

        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_hidden_states: (torch.Tensor) updated RNN hidden states.
        """
        actor_features = self.base(obs)

        if self.use_recurrent_policy:
            actor_features, rnn_hidden_states = self.rnn(
                actor_features, rnn_hidden_states, masks)

        actions, action_log_probs = self.act(actor_features, available_actions,
                                             deterministic)

        return actions, action_log_probs, rnn_hidden_states

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        masks: torch.Tensor,
        available_actions: torch.Tensor = None,
        active_masks: torch.Tensor = None,
        rnn_hidden_states: torch.Tensor = None,
    ):
        """Compute log probability and entropy of given actions.

        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_hidden_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        actor_features = self.base(obs)

        if self.use_recurrent_policy:
            actor_features, rnn_hidden_states = self.rnn(
                actor_features, rnn_hidden_states, masks)

        if self.algorithm_name == 'hatrpo':
            action_log_probs, dist_entropy, action_mu, action_std, all_probs = (
                self.act.evaluate_actions_trpo(
                    actor_features,
                    action,
                    available_actions,
                    active_masks=active_masks
                    if self.use_policy_active_masks else None,
                ))

            return action_log_probs, dist_entropy, action_mu, action_std, all_probs
        else:
            action_log_probs, dist_entropy = self.act.evaluate_actions(
                actor_features,
                action,
                available_actions,
                active_masks=active_masks
                if self.use_policy_active_masks else None,
            )

        return action_log_probs, dist_entropy
