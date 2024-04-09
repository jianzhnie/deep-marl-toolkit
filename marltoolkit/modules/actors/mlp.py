import torch
import torch.nn as nn


# Initialize Policy weights
def weights_init_(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1)
        nn.init.constant_(module.bias, 0)


class MLPActorModel(nn.Module):

    def __init__(
        self,
        input_dim: int = None,
        hidden_dim: int = 64,
        n_actions: int = None,
    ) -> None:
        """Initialize the Actor network.

        Args:
            input_dim (int, optional): obs, include the agent's id and last action,
                    shape: (batch, obs_shape + n_action + n_agents)
            hidden_dim (int, optional): hidden size of the network. Defaults to 64.
            n_actions (int, optional): number of actions. Defaults to None.
        """
        super(MLPActorModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.apply(weights_init_)

    def forward(self, obs: torch.Tensor):
        hid1 = self.relu1(self.fc1(obs))
        hid2 = self.relu2(self.fc2(hid1))
        policy = self.fc3(hid2)
        return policy
