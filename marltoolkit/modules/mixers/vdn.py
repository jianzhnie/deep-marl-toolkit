import torch
import torch.nn as nn


class VDNMixer(nn.Module):

    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, agent_qs, states):
        return torch.sum(agent_qs, dim=2, keepdim=True)
