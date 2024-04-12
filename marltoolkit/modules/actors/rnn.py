import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNActorModel(nn.Module):
    """Because all the agents share the same network,
    input_shape=obs_shape+n_actions+n_agents.

    Args:
        input_dim (int): The input dimension.
        fc_hidden_dim (int): The hidden dimension of the fully connected layer.
        rnn_hidden_dim (int): The hidden dimension of the RNN layer.
        n_actions (int): The number of actions.
    """

    def __init__(
        self,
        input_dim: int = None,
        fc_hidden_dim: int = 64,
        rnn_hidden_dim: int = 64,
        n_actions: int = None,
        **kwargs,
    ) -> None:
        super(RNNActorModel, self).__init__()

        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc1 = nn.Linear(input_dim, fc_hidden_dim)
        self.rnn = nn.GRUCell(input_size=fc_hidden_dim,
                              hidden_size=rnn_hidden_dim)
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

        hidden_state = self.rnn(out, h_in)
        out = self.fc2(hidden_state)  # (batch_size, n_actions)
        return out, hidden_state

    def update(self, model: nn.Module) -> None:
        self.load_state_dict(model.state_dict())
