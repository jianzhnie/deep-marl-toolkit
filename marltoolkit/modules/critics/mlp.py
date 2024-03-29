import torch
import torch.nn as nn
import torch.nn.functional as F


# Initialize Policy weights
def weights_init_(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1)
        nn.init.constant_(module.bias, 0)


class MLPCriticModel(nn.Module):

    def __init__(
        self,
        n_agent: int,
        obs_shape: int = None,
        n_action: int = None,
        hidden_dim: int = 64,
        output_dim: int = 1,
    ):
        super(MLPCriticModel, self).__init__()

        self.n_agent = n_agent
        self.obs_shape = obs_shape
        self.n_action = n_action
        obs_dim = obs_shape * n_agent
        act_dim = n_action * n_agent

        # Set up network layers
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.apply(weights_init_)

    def forward(self, obs: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            obs (torch.Tensor):   (batch_size, T, input_shape)
        Returns:
            q_total (torch.Tensor):  (batch_size, T, 1)
        '''
        inputs = torch.cat([obs, actions], dim=1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q
