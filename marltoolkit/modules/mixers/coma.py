import torch
import torch.nn as nn
import torch.nn.functional as F


class ComaCritic(nn.Module):
    """
    输入:
        当前的状态、当前agent的obs;
        其他agent执行的动作、当前agent的编号对应的one-hot向量;
        所有agent上一个timestep执行的动作;
    输出:
        当前agent的所有可执行动作对应的联合Q值——一个n_actions维向量
    """

    def __init__(
        self,
        input_shape: int,
        critic_dim: int,
        n_actions: int,
    ):
        super(ComaCritic, self).__init__()
        self.fc1 = nn.Linear(input_shape, critic_dim)
        self.fc2 = nn.Linear(critic_dim, critic_dim)
        self.fc3 = nn.Linear(critic_dim, n_actions)

    def forward(self, inputs: torch.Tensor):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q
