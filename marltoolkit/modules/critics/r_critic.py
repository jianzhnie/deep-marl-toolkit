import argparse

import torch
import torch.nn as nn

from marltoolkit.modules.utils.common import MLPBase, RNNLayer
from marltoolkit.modules.utils.popart import PopArt


class R_Critic(nn.Module):
    """Critic network class for MAPPO. Outputs value function predictions given
    centralized input (MAPPO) or local observations (IPPO).

    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args: argparse.Namespace):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self.use_orthogonal = args.use_orthogonal
        self.use_recurrent_policy = args.use_recurrent_policy
        self.rnn_layers = args.rnn_layers
        self.use_popart = args.use_popart
        self.gain = args.gain
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
                self.hidden_size,
                self.hidden_size,
                self.rnn_layers,
                self.use_orthogonal,
            )

        def init_weight(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                init_method(module.weight, gain=self.gain)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        if self.use_popart:
            self.v_out = PopArt(self.hidden_size, 1)
        else:
            self.v_out = nn.Linear(self.hidden_size, 1)

        self.apply(init_weight)

    def forward(
        self,
        cent_obs: torch.Tensor,
        rnn_states: torch.Tensor,
        masks: torch.Tensor,
    ):
        """Compute actions from the given inputs.

        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        critic_features = self.base(cent_obs)
        if self.use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states,
                                                   masks)
        values = self.v_out(critic_features)
        return values, rnn_states
