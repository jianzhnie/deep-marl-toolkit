import numpy as np
import torch
import torch.nn as nn


class VDNMixer(nn.Module):
    """Computes total Q values given agent q values and global states."""

    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor):
        if type(agent_qs) == np.ndarray:
            agent_qs = torch.FloatTensor(agent_qs)
        return torch.sum(agent_qs, dim=2, keepdim=True)
