import torch
import torch.nn as nn
import torch.nn.functional as F


# Initialize Policy weights
def weights_init_(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1)
        nn.init.constant_(module.bias, 0)


class MLPCritic(nn.Module):

    def __init__(
        self,
        input_dim: int = None,
        hidden_dim: int = 64,
        output_dim: int = 1,
    ):
        super(MLPCritic, self).__init__()

        # Set up network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.apply(weights_init_)

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
