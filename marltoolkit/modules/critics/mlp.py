import torch
import torch.nn as nn
import torch.nn.functional as F


# Initialize Policy weights
def weights_init_(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1)
        nn.init.constant_(module.bias, 0)


class CriticModel(nn.Module):

    def __init__(
        self,
        critic_in_dim: int,
        hidden_size: int = 64,
        out_dim: int = 1,
    ):
        super(CriticModel, self).__init__()
        self.fc1 = nn.Linear(critic_in_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_dim)
        self.apply(weights_init_)

    def forward(self, obs_n: torch.Tensor, act_n: torch.Tensor):
        inputs = torch.cat(obs_n + act_n, dim=1)
        hid1 = F.relu(self.fc1(inputs))
        hid2 = F.relu(self.fc2(hid1))
        q_value = self.fc3(hid2)
        q_value = torch.squeeze(q_value, dim=1)
        return q_value
