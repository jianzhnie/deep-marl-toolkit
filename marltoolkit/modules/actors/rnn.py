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
        input: torch.Tensor = None,
        hidden_state: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = F.relu(self.fc1(input), inplace=True)
        if hidden_state is not None:
            h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        else:
            h_in = torch.zeros(out.shape[0],
                               self.rnn_hidden_dim).to(input.device)

        hidden_state = self.rnn(out, h_in)
        out = self.fc2(hidden_state)  # (batch_size, n_actions)
        return out, hidden_state

    def update(self, model: nn.Module) -> None:
        self.load_state_dict(model.state_dict())


class MultiLayerRNNActorModel(nn.Module):
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
        rnn_num_layers: int = 2,
        n_actions: int = None,
        **kwargs,
    ) -> None:
        super(MultiLayerRNNActorModel, self).__init__()

        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers

        self.fc1 = nn.Linear(input_dim, fc_hidden_dim)
        self.rnn = nn.GRU(
            input_size=fc_hidden_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True,
        )
        self.fc2 = nn.Linear(rnn_hidden_dim, n_actions)

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        # make hidden states on same device as model
        return self.fc1.weight.new(self.rnn_num_layers, batch_size,
                                   self.rnn_hidden_dim).zero_()

    def forward(
        self,
        input: torch.Tensor = None,
        hidden_state: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # input: (batch_size, episode_length, obs_dim)
        out = F.relu(self.fc1(input), inplace=True)
        # out:  (batch_size, episode_length, fc_hidden_dim)
        batch_size = input.shape[0]
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size)
        else:
            hidden_state = hidden_state.reshape(
                self.rnn_num_layers, batch_size,
                self.rnn_hidden_dim).to(input.device)

        out, hidden_state = self.rnn(out, hidden_state)
        # out: (batch_size, seq_len, rnn_hidden_dim)
        # hidden_state: (num_ayers, batch_size, rnn_hidden_dim)
        logits = self.fc2(out)
        return logits, hidden_state

    def update(self, model: nn.Module) -> None:
        self.load_state_dict(model.state_dict())


if __name__ == '__main__':
    rnn = nn.GRU(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
    input = torch.randn(32, 512, 10)
    h0 = torch.randn(2, 32, 20)
    output, hn = rnn(input, h0)
    print(output.shape, hn.shape)
    # torch.Size([32, 512, 20]) torch.Size([2, 32, 20])
