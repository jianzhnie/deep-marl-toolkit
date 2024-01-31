import torch
import torch.nn as nn


# Initialize Policy weights
def weights_init_(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1)
        nn.init.constant_(module.bias, 0)


class ActorModel(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_size: int = 64,
        continuous_actions: bool = False,
    ):
        super(ActorModel, self).__init__()
        self.continuous_actions = continuous_actions

        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, act_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        if self.continuous_actions:
            std_hid_size = hidden_size
            self.std_fc = nn.Linear(std_hid_size, act_dim)

    def forward(self, obs: torch.Tensor):
        hid1 = self.relu1(self.fc1(obs))
        hid2 = self.relu2(self.fc2(hid1))
        means = self.fc3(hid2)
        if self.continuous_actions:
            act_std = self.std_fc(hid2)
            return (means, act_std)
        return means
