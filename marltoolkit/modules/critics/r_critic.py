import argparse

import torch
import torch.nn as nn

from marltoolkit.modules.utils.common import MLPBase, RNNLayer
from marltoolkit.modules.utils.popart import PopArt


class R_Critic(nn.Module):
    """Critic network class for MAPPO. Outputs value function predictions given
    centralized input (MAPPO) or local observations (IPPO).

    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param state_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args: argparse.Namespace):
        super(R_Critic, self).__init__()
        self.use_recurrent_policy = args.use_recurrent_policy
        init_method = [nn.init.xavier_uniform_,
                       nn.init.orthogonal_][self.use_orthogonal]
        self.base = MLPBase(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_size,
            use_orthogonal=args.use_orthogonal,
            use_feature_normalization=args.use_feature_normalization,
        )
        if self.use_recurrent_policy:
            self.rnn = RNNLayer(
                args.hidden_size,
                args.hidden_size,
                args.rnn_layers,
                args.use_orthogonal,
            )

        def init_weight(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                init_method(module.weight, gain=args.gain)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        if args.use_popart:
            self.v_out = PopArt(args.hidden_size, 1)
        else:
            self.v_out = nn.Linear(args.hidden_size, 1)

        self.apply(init_weight)

    def forward(
        self,
        state: torch.Tensor,
        masks: torch.Tensor,
        rnn_hidden_states: torch.Tensor,
    ):
        """Compute actions from the given inputs.

        :param state: (np.ndarray / torch.Tensor) global observation inputs into network.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.
        :param rnn_hidden_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_hidden_states: (torch.Tensor) updated RNN hidden states.
        """
        critic_features = self.base(state)
        if self.use_recurrent_policy:
            critic_features, rnn_hidden_states = self.rnn(
                critic_features, rnn_hidden_states, masks)
        values = self.v_out(critic_features)
        return values, rnn_hidden_states
