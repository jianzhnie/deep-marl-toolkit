'''
Author: jianzhnie
LastEditors: jianzhnie
Description: RLToolKit is a flexible and high-efficient reinforcement learning framework.
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNActor(nn.Module):

    def __init__(
        self,
        input_shape: int = None,
        rnn_hidden_dim: int = 64,
        n_actions: int = None,
    ):
        super(RNNActor, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim

        self.fc1 = nn.Linear(input_shape, rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        self.fc2 = nn.Linear(rnn_hidden_dim, n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self,
                inputs: torch.Tensor = None,
                hidden_state: torch.Tensor = None):
        x = F.relu(self.fc1(inputs), inplace=True)
        if hidden_state is not None:
            h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)

        h = self.rnn(x, h_in)
        q = self.fc2(h)  # (batch_size, n_actions)
        return q, h

    def update(self, model: nn.Module):
        self.load_state_dict(model.state_dict())


class RNNFeatureAgent(nn.Module):
    """Identical to rnn_agent, but does not compute value/probability for each
    action, only the hidden state."""

    def __init__(self, input_shape, rnn_hidden_dim: int = 64):
        super(RNNFeatureAgent, self).__init__()

        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc1 = nn.Linear(input_shape, rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = nn.functional.relu(self.fc1(inputs))
        h = self.rnn(x, hidden_state.reshape(-1, self.rnn_hidden_dim))
        return None, h


class RNNNSAgent(nn.Module):

    def __init__(
        self,
        n_agents: int,
        input_shape: int = None,
        rnn_hidden_dim: int = 64,
        n_actions: int = None,
    ):
        super(RNNNSAgent, self).__init__()
        self.n_agents = n_agents
        self.input_shape = input_shape
        self.agents: List[RNNActor] = nn.ModuleList([
            RNNActor(
                input_shape=input_shape,
                rnn_hidden_dim=rnn_hidden_dim,
                n_actions=n_actions,
            ) for _ in range(self.n_agents)
        ])

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.cat([agent.init_hidden() for agent in self.agents])

    def forward(self,
                inputs: torch.Tensor = None,
                hidden_state: torch.Tensor = None):
        hiddens = []
        qs = []
        if inputs.size(0) == self.n_agents:
            for i in range(self.n_agents):
                q, h = self.agents[i](inputs[i].unsqueeze(0), hidden_state[:,
                                                                           i])
                hiddens.append(h)
                qs.append(q)
            return torch.cat(qs), torch.cat(hiddens).unsqueeze(0)
        else:
            for i in range(self.n_agents):
                inputs = inputs.view(-1, self.n_agents, self.input_shape)
                q, h = self.agents[i](inputs[:, i], hidden_state[:, i])
                hiddens.append(h.unsqueeze(1))
                qs.append(q.unsqueeze(1))
            return torch.cat(
                qs, dim=-1).view(-1, q.size(-1)), torch.cat(
                    hiddens, dim=1)
