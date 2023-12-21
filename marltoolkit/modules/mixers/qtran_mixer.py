'''
Author: jianzhnie
LastEditors: jianzhnie
Description: RLToolKit is a flexible and high-efficient reinforcement learning framework.
Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
'''

import torch
import torch.nn as nn


class QTransModel(nn.Module):
    """Joint action-value network
       输入: state、所有 agent 的 hidden_states、其他 agent 的动作，
       输出: 所有动作对应的联合Q值
    """

    def __init__(self,
                 n_agents: int = None,
                 n_actions: int = None,
                 state_dim: int = None,
                 rnn_hidden_dim: int = 32,
                 mixing_embed_dim: int = 32,
                 qtran_arch: str = 'qtran_paper',
                 network_size: str = 'small'):
        super(QTransModel, self).__init__()

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.qtran_arch = qtran_arch
        self.network_size = network_size

        # action_encoding 对输入的每个 agent的 hidden_states 和动作进行编码，
        # 从而将所有 agents 的 hidden_states 和动作相加得到近似的联合 hidden_states 和动作
        ae_input = self.rnn_hidden_dim + self.n_actions
        self.action_encoding = nn.Sequential(
            nn.Linear(ae_input, ae_input), nn.ReLU(inplace=True),
            nn.Linear(ae_input, ae_input))

        # Q(s,u)
        if self.qtran_arch == 'coma_critic':
            # Q takes [state, u] as input
            q_input_size = state_dim + (n_agents * n_actions)
        elif self.qtran_arch == 'qtran_paper':
            # Q takes [state, agent_action_observation_encodings]
            # 编码求和之后输入 state、所有 agent 的 hidden_states 和动作之和
            q_input_size = state_dim + rnn_hidden_dim + n_actions
        else:
            raise Exception('{} is not a valid QTran architecture'.format(
                self.qtran_arch))

        if self.network_size == 'small':
            self.Q = nn.Sequential(
                nn.Linear(q_input_size, mixing_embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(mixing_embed_dim, mixing_embed_dim),
                nn.ReLU(inplace=True), nn.Linear(mixing_embed_dim, 1))

            # V(s)
            self.V = nn.Sequential(
                nn.Linear(state_dim, mixing_embed_dim), nn.ReLU(inplace=True),
                nn.Linear(mixing_embed_dim, mixing_embed_dim),
                nn.ReLU(inplace=True), nn.Linear(mixing_embed_dim, 1))

        elif self.network_size == 'big':
            self.Q = nn.Sequential(
                nn.Linear(q_input_size, mixing_embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(mixing_embed_dim, mixing_embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(mixing_embed_dim, mixing_embed_dim),
                nn.ReLU(inplace=True), nn.Linear(mixing_embed_dim, 1))
            # V(s)
            self.V = nn.Sequential(
                nn.Linear(self.state_dim, mixing_embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(mixing_embed_dim, mixing_embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(mixing_embed_dim, mixing_embed_dim),
                nn.ReLU(inplace=True), nn.Linear(mixing_embed_dim, 1))

        else:
            raise Exception('{} is not a valid QTran architecture'.format(
                self.network_size))

    # 因为所有时刻所有 agent 的 hidden_states 在之前已经计算好了，
    # 所以联合 Q 值可以一次计算所有 transition 的，不需要一条一条计算。
    def forward(self, state, hidden_states, actions=None):
        '''
        Args:
            state (torch.Tensor):          (batch_size, T, state_dim)
            hidden_states (torch.Tensor):   (batch_size, T, n_agents, n_actions)
            actions (torch.Tensor):         (batch_size, T, n_agents, n_actions)
        Returns:
            q_total (torch.Tensor):  (batch_size, T, 1)
            v_total (torch.Tensor):  (batch_size, T, 1)
        '''
        batch_size = state.size(0)
        episode_len = state.size(1)

        # state : (batch_size * T, state_dim)
        state = state.reshape(-1, self.state_dim)

        if self.qtran_arch == 'coma_critic':
            # It will arrive as (bs, ts, agents, actions), we need to reshape it
            actions = actions.reshape(-1, self.n_agents * self.n_actions)

            inputs = torch.cat([state, actions], dim=1)

        elif self.qtran_arch == 'qtran_paper':
            # It will arrive as (bs, ts, agents, actions), we need to reshape it
            actions = actions.reshape(batch_size * episode_len, self.n_agents,
                                      self.n_actions)

            hidden_states = hidden_states.reshape(batch_size * episode_len,
                                                  self.n_agents, -1)
            state_action = torch.cat([hidden_states, actions], dim=2)

            state_action_reshape = state_action.reshape(
                batch_size * episode_len, self.n_agents, -1)

            state_action_encoding = self.action_encoding(state_action_reshape)

            state_action_encoding = state_action_encoding.reshape(
                batch_size * episode_len, self.n_agents, -1)
            # Sum across agents
            state_action_encoding = state_action_encoding.sum(dim=1)
            inputs = torch.cat([state, state_action_encoding], dim=1)

        q_outputs = self.Q(inputs)
        v_outputs = self.V(state)

        return q_outputs, v_outputs


class QTranBase(nn.Module):
    """Joint action-value network
       输入: state、所有 agent 的 hidden_states、其他 agent 的动作，
       输出: 所有动作对应的联合Q值
    """

    def __init__(self,
                 n_agents: int = None,
                 n_actions: int = None,
                 state_dim: int = None,
                 rnn_hidden_dim: int = 32,
                 mixing_embed_dim: int = 32):
        super(QTranBase, self).__init__()

        self.q = QtranQ(n_agents, n_actions, state_dim, rnn_hidden_dim,
                        mixing_embed_dim)

        self.v = QtranV(n_agents, n_actions, state_dim, rnn_hidden_dim,
                        mixing_embed_dim)

    def forward(self, state, hidden_states, actions=None):
        '''
        Args:
            state (torch.Tensor):          (batch_size, T, state_dim)
            hidden_states (torch.Tensor):   (batch_size, T, n_agents, n_actions)
            actions (torch.Tensor):         (batch_size, T, n_agents, n_actions)
        Returns:
            q_total (torch.Tensor):  (batch_size, T, 1)
            v_total (torch.Tensor):  (batch_size, T, 1)
        '''

        # (batch_size * T, 1)
        q_total = self.q(state, hidden_states, actions)
        v_total = self.v(state, hidden_states)
        return q_total, v_total


class QtranQ(nn.Module):
    """Joint action-value network,
       输入: state, 所有 agent 的 hidden_states,所有agent的动作,
       输出: 对应的联合Q值

    Steps:
        - action_encoder 对输入的每个 agent 的 hidden_states 和动作进行编码，
        - 将所有 agents 的 hidden_states 和动作相加得到近似的联合hidden_states和动作
    """

    def __init__(self,
                 n_agents: int = None,
                 n_actions: int = None,
                 state_dim: int = None,
                 rnn_hidden_dim: int = 32,
                 mixing_embed_dim: int = 32):
        super(QtranQ, self).__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        ae_input = rnn_hidden_dim + n_actions
        self.hidden_action_encoder = nn.Sequential(
            nn.Linear(ae_input, ae_input), nn.ReLU(inplace=True),
            nn.Linear(ae_input, ae_input))

        q_input = state_dim + n_actions + rnn_hidden_dim
        self.q = nn.Sequential(
            nn.Linear(q_input, mixing_embed_dim), nn.ReLU(inplace=True),
            nn.Linear(mixing_embed_dim, mixing_embed_dim),
            nn.ReLU(inplace=True), nn.Linear(mixing_embed_dim, 1))

    def forward(self, state, hidden_states, actions):
        '''
        Args:
            state (torch.Tensor):          (batch_size, T, state_dim)
            hidden_states (torch.Tensor):   (batch_size, T, n_agents, rnn_hidden_dim)
            actions (torch.Tensor):         (batch_size, T, n_agents, n_actions)
        Returns:
            q_total (torch.Tensor):  (batch_size, T, 1)
        '''
        batch_size, episode_len, n_agents, _ = hidden_states.shape
        # shape : (batch_size * T, state_dim)
        state = state.reshape(-1, self.state_dim)

        # shape : (batch_size, T,  n_agents, rnn_hidden_dim + n_actions)
        hidden_actions = torch.cat([hidden_states, actions], dim=-1)
        # shape : (batch_size * T * n_agents, rnn_hidden_dim + n_actions)
        hidden_actions = hidden_actions.reshape(
            -1, self.rnn_hidden_dim + self.n_actions)
        # shape : (batch_size * T * n_agents, rnn_hidden_dim + n_actions)
        hidden_actions_encoding = self.hidden_action_encoder(hidden_actions)
        # 变回 n_agents 维度用于求和
        # shape: (batch_size * T, n_agents, rnn_hidden_dim + n_actions)
        hidden_actions_encoding = hidden_actions_encoding.reshape(
            batch_size * episode_len, self.n_agents, -1)
        # shape: (batch_size * T, rnn_hidden_dim + n_actions)
        hidden_actions_encoding = hidden_actions_encoding.sum(dim=-2)
        # 编码求和之后输入 state、所有agent的hidden_states 和动作之和
        # shape: (batch_size * T, state + rnn_hidden_dim + n_actions)
        inputs = torch.cat([state, hidden_actions_encoding], dim=-1)
        # shape: (batch_size * T, 1)
        q_total = self.q(inputs)
        return q_total


class QtranV(nn.Module):
    """
    输入: 当前的 state 与所有 agent 的 hidden_states,
    输出: V值

    Steps:
        - hidden_encoder 对输入的每个 agent 的 hidden_states 编码，
        - 将所有 agents 的 hidden_states 相加得到近似的联合 hidden_states
    """

    def __init__(self,
                 n_agents: int = None,
                 n_actions: int = None,
                 state_dim: int = None,
                 rnn_hidden_dim: int = 32,
                 mixing_embed_dim: int = 32):
        super(QtranV, self).__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        self.hidden_encoder = nn.Sequential(
            nn.Linear(rnn_hidden_dim, rnn_hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(rnn_hidden_dim, rnn_hidden_dim))

        v_input = state_dim + rnn_hidden_dim
        self.v = nn.Sequential(
            nn.Linear(v_input, mixing_embed_dim), nn.ReLU(inplace=True),
            nn.Linear(mixing_embed_dim, mixing_embed_dim),
            nn.ReLU(inplace=True), nn.Linear(mixing_embed_dim, 1))

    def forward(self, state: torch.Tensor,
                hidden_states: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            state (torch.Tensor):          (batch_size, T, state_dim)
            hidden_states (torch.Tensor):   (batch_size, T, n_agents, rnn_hidden_dim)
        Returns:
            v_total (torch.Tensor):  (batch_size, T, 1)
        '''
        batch_size, episode_len, n_agents, _ = hidden_states.shape
        # state : (batch_size * T, state_dim)
        state = state.reshape(batch_size * episode_len, -1)

        # (batch_size * T * n_agents, rnn_hidden_dim)
        hidden_states = hidden_states.reshape(-1, self.rnn_hidden_dim)
        # (batch_size * T * n_agents, rnn_hidden_dim)
        hidden_encoding = self.hidden_encoder(hidden_states)
        # (batch_size * T * n_agents, rnn_hidden_dim) ==>
        # (batch_size * T, n_agents, rnn_hidden_dim)
        hidden_encoding = hidden_encoding.reshape(batch_size * episode_len,
                                                  n_agents, -1)
        # (batch_size * T, rnn_hidden_dim)
        hidden_encoding = hidden_encoding.sum(dim=-2)
        # 编码求和之后输入state、所有agent的hidden_states之和
        # (batch_size * T, state + rnn_hidden_dim)
        inputs = torch.cat([state, hidden_encoding], dim=-1)
        # (batch_size * T, 1)
        v_total = self.v(inputs)
        return v_total
