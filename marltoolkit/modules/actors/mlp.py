import torch
import torch.nn as nn


# Initialize Policy weights
def weights_init_(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1)
        nn.init.constant_(module.bias, 0)


class MLPActorModel(nn.Module):

    def __init__(self, obs_shape: int, n_actions: int, hidden_size: int = 64):
        super(MLPActorModel, self).__init__()

        self.fc1 = nn.Linear(obs_shape, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.apply(weights_init_)

    def forward(self, obs: torch.Tensor):
        hid1 = self.relu1(self.fc1(obs))
        hid2 = self.relu2(self.fc2(hid1))
        policy = self.fc3(hid2)
        return policy
