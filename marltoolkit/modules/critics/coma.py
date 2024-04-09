import torch.nn as nn

from marltoolkit.modules.actors.mlp import MLPActorModel
from marltoolkit.modules.critics.mlp import MLPCriticModel


class ComaModel(nn.Module):

    def __init__(
        self,
        num_agents: int = None,
        n_actions: int = None,
        obs_shape: int = None,
        state_shape: int = None,
        hidden_dim: int = 64,
        **kwargs,
    ):
        super(ComaModel, self).__init__()

        self.num_agents = num_agents
        self.obs_shape = obs_shape
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.actions_onehot_shape = self.n_actions * self.num_agents
        actor_input_dim = self._get_actor_input_dim()
        critic_input_dim = self._get_critic_input_dim()

        # Set up network layers
        self.actor_model = MLPActorModel(input_dim=actor_input_dim,
                                         hidden_dim=hidden_dim,
                                         n_actions=n_actions)
        self.critic_model = MLPCriticModel(input_dim=critic_input_dim,
                                           hidden_dim=hidden_dim,
                                           output_dim=1)

    def policy(self, obs, hidden_state):
        return self.actor_model(obs, hidden_state)

    def value(self, inputs):
        return self.critic_model(inputs)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()

    def _get_actor_input_dim(self):
        # observation
        input_dim = self.obs_shape
        # agent id
        input_dim += self.num_agents
        # actions and last actions
        input_dim += self.actions_onehot_shape * self.num_agents * 2
        return input_dim

    def _get_critic_input_dim(self):
        input_dim = self.state_shape  # state: 48 in 3m map
        input_dim += self.obs_shape  # obs: 30 in 3m map
        input_dim += self.num_agents  # agent_id: 3 in 3m map
        input_dim += (
            self.n_actions * self.num_agents * 2
        )  # all agents' action and last_action (one-hot): 54 in 3m map
        return input_dim  # 48 + 30+ 3 = 135
