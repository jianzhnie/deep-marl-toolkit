import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):

    def __init__(
        self,
        input_dim: int = None,
        fc_hidden_dim: int = 64,
        num_rnn_layers: int = 1,
        rnn_hidden_dim: int = 64,
        dropout: float = 0.0,
        action_dim: int = None,
    ) -> None:
        super(RNNModel, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc1 = nn.Linear(input_dim, fc_hidden_dim)
        self.rnn = nn.GRU(
            input_size=rnn_hidden_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc2 = nn.Linear(rnn_hidden_dim, action_dim)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self,
                inputs: torch.Tensor = None,
                hidden_state: torch.Tensor = None):
        x = F.relu(self.fc1(inputs), inplace=True)
        if hidden_state is not None:
            h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)

        h = self.rnn(x, h_in)
        q = self.fc2(h)  # (batch_size, n_actions)
        return q, h

    def update(self, model: nn.Module) -> None:
        self.load_state_dict(model.state_dict())
