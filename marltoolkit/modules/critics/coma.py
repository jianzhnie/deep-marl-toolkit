import torch
import torch.nn as nn
import torch.nn.functional as F


class COMACritic(nn.Module):

    def __init__(
        self,
        n_agents: int = None,
        state_shape: int = None,
        obs_shape: int = None,
        n_actions: int = None,
        hidden_dim: int = 64,
        device: str = 'cpu',
    ):
        super(COMACritic, self).__init__()

        self.n_agents = n_agents
        self.state_shape = state_shape
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.actions_onehot_shape = self.n_actions * self.n_agents
        input_shape = self._get_input_shape()
        self.device = device

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.n_actions)

    def forward(self, state, obs, actions_onehot, t=None):
        inputs = self._build_inputs(state, obs, actions_onehot, t=t)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _build_inputs(self, state, obs, actions_onehot, t=None):
        bs = state.shape[0]
        max_t = state.shape[1] if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)
        inputs = []
        # state
        inputs.append(state[:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))
        # observation
        inputs.append(obs[:, ts])
        # actions (masked out by agent)
        actions = actions_onehot[:,
                                 ts].view(bs, max_t, 1,
                                          -1).repeat(1, 1, self.n_agents, 1)
        agent_mask = (1 - torch.eye(self.n_agents, device=self.device))
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions).view(
            self.n_agents, -1)
        inputs.append(actions * agent_mask.unsqueeze(0).unsqueeze(0))

        # last actions
        if t == 0:
            inputs.append(
                torch.zeros_like(actions_onehot[:, 0:1]).view(
                    bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
        elif isinstance(t, int):
            inputs.append(actions_onehot[:, slice(t - 1, t)].view(
                bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
        else:
            last_actions = torch.cat([
                torch.zeros_like(actions_onehot[:, 0:1]),
                actions_onehot[:, :-1]
            ],
                                     dim=1)
            last_actions = last_actions.view(bs, max_t, 1, -1).repeat(
                1, 1, self.n_agents, 1)
            inputs.append(last_actions)

        inputs.append(
            torch.eye(self.n_agents,
                      device=self.device).unsqueeze(0).unsqueeze(0).expand(
                          bs, max_t, -1, -1))

        inputs = torch.cat(
            [x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self):
        # state
        input_shape = self.state_shape
        # observation
        input_shape += self.obs_shape
        # actions and last actions
        input_shape += self.actions_onehot_shape * self.n_agents * 2
        # agent id
        input_shape += self.n_agents
        return input_shape
