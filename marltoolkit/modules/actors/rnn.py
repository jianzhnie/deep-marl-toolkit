import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNActor(nn.Module):

    def __init__(
        self,
        input_dim: int = None,
        fc_hidden_dim: int = 64,
        num_rnn_layers: int = 1,
        rnn_hidden_dim: int = 64,
        n_actions: int = None,
        **kwargs,
    ) -> None:
        super(RNNActor, self).__init__()
        self.num_rnn_layers = num_rnn_layers
        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc1 = nn.Linear(input_dim, fc_hidden_dim)
        self.rnn = nn.GRUCell(
            input_size=fc_hidden_dim,
            hidden_size=rnn_hidden_dim,
        )
        self.fc2 = nn.Linear(rnn_hidden_dim, n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(
        self,
        inputs: torch.Tensor = None,
        hidden_state: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        out = F.relu(self.fc1(inputs), inplace=True)
        if hidden_state is not None:
            h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        else:
            h_in = torch.zeros(out.shape[0],
                               self.rnn_hidden_dim).to(inputs.device)

        hidden_states = self.rnn(out, h_in)
        q_value = self.fc2(hidden_states)  # (batch_size, n_actions)
        return q_value, hidden_states

    def update(self, model: nn.Module) -> None:
        self.load_state_dict(model.state_dict())
