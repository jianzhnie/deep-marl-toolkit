import torch
import torch.nn as nn
import torch.nn.functional as F


# Initialize Policy weights
def weights_init_(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1)
        nn.init.constant_(module.bias, 0)


class MLPCriticModel(nn.Module):
    """MLP Critic Network.

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        input_dim: int = None,
        hidden_dim: int = 64,
        output_dim: int = 1,
    ) -> None:
        super(MLPCriticModel, self).__init__()

        # Set up network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.apply(weights_init_)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor):
        Returns:
            q_total (torch.Tensor):
        """
        hidden = F.relu(self.fc1(inputs))
        hidden = F.relu(self.fc2(hidden))
        qvalue = self.fc3(hidden)
        return qvalue
