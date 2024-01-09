import torch
import torch.nn as nn
import torch.nn.functional as F


class ACCritic(nn.Module):

    def __init__(
        self,
        n_agents: int = None,
        input_dim: int = None,
        hidden_dim: int = 64,
        n_actions: int = None,
    ):
        super(ACCritic, self).__init__()

        self.n_agents = n_agents
        self.input_shape = input_dim
        self.n_actions = n_actions

        # Set up network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            inputs (torch.Tensor):   (batch_size, T, input_shape)
        Returns:
            q_total (torch.Tensor):  (batch_size, T, 1)
        '''
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q
