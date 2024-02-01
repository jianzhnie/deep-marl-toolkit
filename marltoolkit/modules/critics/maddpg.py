import torch
import torch.nn as nn
import torch.nn.functional as F


class MADDPGCritic(nn.Module):

    def __init__(
        self,
        n_agents,
        state_shape,
        obs_shape,
        hidden_dim,
        n_actions,
        obs_agent_id: bool,
        obs_last_action: bool,
        obs_individual_obs: bool,
    ):
        super(MADDPGCritic, self).__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.obs_shape = obs_shape
        self.obs_individual_obs = obs_individual_obs
        self.obs_agent_id = obs_agent_id
        self.obs_last_action = obs_last_action

        self.input_shape = self._get_input_shape(
        ) + self.n_actions * self.n_agents
        if self.obs_last_action:
            self.input_shape += self.n_actions

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, inputs, actions):
        inputs = torch.cat((inputs, actions), dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _get_input_shape(self):
        # state_shape
        input_shape = self.state_shape
        # whether to add the individual observation
        if self.obs_individual_obs:
            input_shape += self.obs_shape
        # agent id
        if self.obs_agent_id:
            input_shape += self.n_agents
        return input_shape
