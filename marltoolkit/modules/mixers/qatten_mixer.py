import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QattenMixer(nn.Module):

    def __init__(self,
                 n_agents: int = None,
                 state_dim: int = None,
                 agent_own_state_size: int = None,
                 n_query_embedding_layer1: int = 64,
                 n_query_embedding_layer2: int = 32,
                 n_key_embedding_layer1: int = 32,
                 n_head_embedding_layer1: int = 64,
                 n_head_embedding_layer2: int = 4,
                 n_attention_head: int = 4,
                 n_constrant_value: int = 32):
        super(QattenMixer, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.agent_own_state_size = agent_own_state_size

        self.n_query_embedding_layer1 = n_query_embedding_layer1
        self.n_query_embedding_layer2 = n_query_embedding_layer2
        self.n_key_embedding_layer1 = n_key_embedding_layer1
        self.n_head_embedding_layer1 = n_head_embedding_layer1
        self.n_head_embedding_layer2 = n_head_embedding_layer2
        self.n_attention_head = n_attention_head
        self.n_constrant_value = n_constrant_value

        self.query_embedding_layers = nn.ModuleList()
        for i in range(self.n_attention_head):
            self.query_embedding_layers.append(
                nn.Sequential(
                    nn.Linear(state_dim, n_query_embedding_layer1),
                    nn.ReLU(inplace=True),
                    nn.Linear(n_query_embedding_layer1,
                              n_query_embedding_layer2)))

        self.key_embedding_layers = nn.ModuleList()
        for i in range(n_attention_head):
            self.key_embedding_layers.append(
                nn.Linear(self.agent_own_state_size, n_key_embedding_layer1))

        self.scaled_product_value = np.sqrt(n_query_embedding_layer2)

        self.head_embedding_layer = nn.Sequential(
            nn.Linear(state_dim, n_head_embedding_layer1),
            nn.ReLU(inplace=True),
            nn.Linear(n_head_embedding_layer1, n_head_embedding_layer2))

        self.constrant_value_layer = nn.Sequential(
            nn.Linear(state_dim, n_constrant_value), nn.ReLU(inplace=True),
            nn.Linear(n_constrant_value, 1))

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor):
        '''
        Args:
            agent_qs (torch.Tensor): (batch_size, T, n_agents)
            states (torch.Tensor):   (batch_size, T, state_shape)
        Returns:
            q_total (torch.Tensor):  (batch_size, T, 1)
        '''
        bs = agent_qs.size(0)
        # states : (batch_size * T, state_shape)
        states = states.reshape(-1, self.state_dim)
        # agent_qs: (batch_size * T, 1, n_agents)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        us = self._get_us(states)

        q_lambda_list = []
        for i in range(self.n_attention_head):
            state_embedding = self.query_embedding_layers[i](states)
            u_embedding = self.key_embedding_layers[i](us)

            # shape: [batch_size * T, 1, state_dim]
            state_embedding = state_embedding.reshape(
                -1, 1, self.n_query_embedding_layer2)
            # shape: [batch_size * T, state_dim, n_agent]
            u_embedding = u_embedding.reshape(-1, self.n_agents,
                                              self.n_key_embedding_layer1)
            u_embedding = u_embedding.permute(0, 2, 1)

            # shape: [batch_size * T, 1, n_agent]
            raw_lambda = torch.matmul(state_embedding,
                                      u_embedding) / self.scaled_product_value
            q_lambda = F.softmax(raw_lambda, dim=-1)

            q_lambda_list.append(q_lambda)

        # shape: [batch_size * T, n_attention_head, n_agent]
        q_lambda_list = torch.stack(q_lambda_list, dim=1).squeeze(-2)

        # shape: [batch_size * T, n_agent, n_attention_head]
        q_lambda_list = q_lambda_list.permute(0, 2, 1)

        # shape: [batch_size * T, 1, n_attention_head]
        q_h = torch.matmul(agent_qs, q_lambda_list)

        if self.type == 'weighted':
            # shape: [batch_size * T, n_attention_head]
            w_h = torch.abs(self.head_embedding_layer(states))
            # shape: [batch_size * T, n_attention_head, 1]
            w_h = w_h.reshape(-1, self.n_head_embedding_layer2, 1)

            # shape: [batch_size * T, 1,1]
            sum_q_h = torch.matmul(q_h, w_h)
            # shape: [batch_size * T, 1]
            sum_q_h = sum_q_h.reshape(-1, 1)
        else:
            # shape: [-1, 1]
            sum_q_h = q_h.sum(-1)
            sum_q_h = sum_q_h.reshape(-1, 1)

        c = self.constrant_value_layer(states)
        q_tot = sum_q_h + c
        q_tot = q_tot.view(bs, -1, 1)
        return q_tot

    def _get_us(self, states):
        agent_own_state_size = self.agent_own_state_size
        with torch.no_grad():
            us = states[:, :agent_own_state_size * self.n_agents].reshape(
                -1, agent_own_state_size)
        return us
