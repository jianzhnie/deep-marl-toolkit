from typing import Any

import torch
import torch.nn as nn


class MLPBase(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation: str = 'relu',
        use_orthogonal: bool = False,
        use_feature_normalization: bool = False,
    ) -> None:
        super(MLPBase, self).__init__()

        use_relu = 1 if activation == 'relu' else 0
        active_func = [nn.ReLU(), nn.Tanh()][use_relu]
        if use_orthogonal:
            init_method = nn.init.orthogonal_
        else:
            init_method = nn.init.xavier_uniform_

        gain = nn.init.calculate_gain(['tanh', 'relu'][use_relu])
        self.use_orthogonal = use_orthogonal
        self.use_feature_normalization = use_feature_normalization
        if use_feature_normalization:
            self.feature_norm = nn.LayerNorm(input_dim)

        def init_weight(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                init_method(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.fc1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), active_func,
                                 nn.LayerNorm(hidden_dim))
        self.fc2 = nn.Sequential(nn.Linear(input_dim, hidden_dim), active_func,
                                 nn.LayerNorm(hidden_dim))
        self.apply(init_weight)

    def forward(self, input: torch.Tensor):
        if self.use_feature_normalization:
            output = self.feature_norm(input)

        output = self.fc1(output)
        output = self.fc2(output)

        return output


class RNNBase(nn.Module):

    def __init__(
        self,
        input_dim: int,
        rnn_hidden_dim: int,
        rnn_layers: int,
        use_orthogonal: bool = True,
    ) -> None:
        super(RNNBase, self).__init__()
        self.rnn_layers = rnn_layers
        self.use_orthogonal = use_orthogonal

        self.rnn = nn.GRU(input_dim, rnn_hidden_dim, num_layers=rnn_layers)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self.use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.layer_norm = nn.LayerNorm(rnn_hidden_dim)

    def forward(
        self,
        input: torch.Tensor,
        hidden_state: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[Any, torch.Tensor]:
        if input.size(0) == hidden_state.size(0):
            input = input.unsqueeze(0)
            hidden_state = (hidden_state * masks.repeat(
                1, self.rnn_layers).unsqueeze(-1).transpose(0, 1).contiguous())
            output, hidden_state = self.rnn(input, hidden_state)
            output = output.squeeze(0)
            hidden_state = hidden_state.transpose(0, 1)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hidden_state.size(0)
            T = int(input.size(0) / N)
            # unflatten
            input = input.view(T, N, input.size(1))
            # Same deal with masks
            masks = masks.view(T, N)
            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = (masks[1:] == 0.0).any(
                dim=-1).nonzero().squeeze().cpu()

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hidden_state = hidden_state.transpose(0, 1)

            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                temp = (hidden_state * masks[start_idx].view(1, -1, 1).repeat(
                    self.rnn_layers, 1, 1)).contiguous()
                rnn_scores, hidden_state = self.rnn(input[start_idx:end_idx],
                                                    temp)
                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            input = torch.cat(outputs, dim=0)

            # flatten
            input = input.reshape(T * N, -1)
            hidden_state = hidden_state.transpose(0, 1)

        output = self.layer_norm(input)
        return output, hidden_state
