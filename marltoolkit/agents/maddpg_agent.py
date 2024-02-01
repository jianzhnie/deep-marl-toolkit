import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import Space

from marltoolkit.utils import check_model_method

from .base_agent import BaseAgent


class MaddpgAgent(BaseAgent):
    """MADDPG algorithm.

    Args:
        model (parl.Model): forward network of actor and critic.
                            The function get_actor_params() of model should be implemented.
        agent_index (int): index of agent, in multiagent env
        action_space (list): action_space, gym space
        gamma (float): discounted factor for reward computation.
        tau (float): decay coefficient when updating the weights of self.target_model with self.model
        critic_lr (float): learning rate of the critic model
        actor_lr (float): learning rate of the actor model
    """

    def __init__(
        self,
        actor_model: nn.Module,
        critic_model: nn.Module,
        agent_index: int = None,
        action_space: Space = None,
        gamma: float = None,
        tau: float = None,
        actor_lr: float = None,
        critic_lr: float = None,
        device: str = None,
    ):

        # checks
        check_model_method(actor_model, 'policy', self.__class__.__name__)
        check_model_method(critic_model, 'value', self.__class__.__name__)

        assert isinstance(agent_index, int)
        assert isinstance(action_space, Space)
        assert isinstance(gamma, float)
        assert isinstance(tau, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)

        self.continuous_actions = False
        if not len(action_space) == 0 and hasattr(action_space[0], 'high') \
                and not hasattr(action_space[0], 'num_discrete_space'):
            self.continuous_actions = True

        self.agent_index = agent_index
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.global_steps = 0
        self.device = device

        self.actor_model = actor_model
        self.critic_model = critic_model
        self.actor_target = copy.deepcopy(actor_model)
        self.critic_target = copy.deepcopy(critic_model)
        self.actor_optimizer = torch.optim.Adam(
            lr=self.actor_lr, params=self.actor_model.parameters)
        self.critic_optimizer = torch.optim.Adam(
            lr=self.critic_lr, params=self.critic_model.parameters)

    def sample(self, obs, use_target_model=False):
        """use the policy model to sample actions.

        Args:
            obs (torch tensor): observation, shape([B] + shape of obs_n[agent_index])
            use_target_model (bool): use target_model or not

        Returns:
            act (torch tensor): action, shape([B] + shape of act_n[agent_index]),
                noted that in the discrete case we take the argmax along the last axis as action
        """
        obs = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)

        if use_target_model:
            policy = self.actor_target(obs)
        else:
            policy = self.actor_model(obs)

        # add noise for action exploration
        if self.continuous_actions:
            random_normal = torch.randn(size=policy[0].shape).to(self.device)
            action = policy[0] + torch.exp(policy[1]) * random_normal
            action = torch.tanh(action)
        else:
            uniform = torch.rand_like(policy)
            soft_uniform = torch.log(-1.0 * torch.log(uniform)).to(self.device)
            action = F.softmax(policy - soft_uniform, dim=-1)

        action = action.detach().cpu().numpy().flatten()
        return action

    def predict(self, obs: torch.Tensor):
        """use the policy model to predict actions.

        Args:
            obs (torch tensor): observation, shape([B] + shape of obs_n[agent_index])

        Returns:
            act (torch tensor): action, shape([B] + shape of act_n[agent_index]),
                noted that in the discrete case we take the argmax along the last axis as action
        """
        obs = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)
        policy = self.actor_model(obs)
        if self.continuous_actions:
            action = policy[0]
            action = torch.tanh(action)
        else:
            action = torch.argmax(policy, dim=-1)

        action = action.detach().cpu().numpy().flatten()
        return action

    def q_value(self, obs_n, act_n, use_target_model=False):
        """use the value model to predict Q values.

        Args:
            obs_n (list of torch tensor): all agents' observation, len(agent's num) + shape([B] + shape of obs_n)
            act_n (list of torch tensor): all agents' action, len(agent's num) + shape([B] + shape of act_n)
            use_target_model (bool): use target_model or not

        Returns:
            Q (torch tensor): Q value of this agent, shape([B])
        """
        if use_target_model:
            return self.critic_target.value(obs_n, act_n)
        else:
            return self.critic_model.value(obs_n, act_n)

    def agent_learn(self, obs_n, act_n, target_q):
        """update actor and critic model with MADDPG algorithm."""
        self.global_steps += 1

    def learn(self, obs_n, act_n, target_q):
        """update actor and critic model with MADDPG algorithm."""
        acotr_loss = self.actor_learn(obs_n, act_n)
        critic_loss = self.critic_learn(obs_n, act_n, target_q)
        return acotr_loss, critic_loss

    def actor_learn(self, obs_n, act_n):
        i = self.agent_index

        sample_this_action = self.sample(obs_n[i])
        action_input_n = act_n + []
        action_input_n[i] = sample_this_action
        eval_q = self.q_value(obs_n, action_input_n)
        act_cost = torch.mean(-1.0 * eval_q)

        this_policy = self.actor_model.policy(obs_n[i])
        # when continuous, 'this_policy' will be a tuple with two element: (mean, std)
        if self.continuous_actions:
            this_policy = torch.cat(this_policy, dim=-1)
        act_reg = torch.mean(torch.square(this_policy))

        cost = act_cost + act_reg * 1e-3

        self.actor_optimizer.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), 0.5)
        self.actor_optimizer.step()
        return cost

    def critic_learn(self, obs_n, act_n, target_q):
        pred_q = self.q_value(obs_n, act_n)
        cost = F.mse_loss(pred_q, target_q)

        self.critic_optimizer.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), 0.5)
        self.critic_optimizer.step()
        return cost

    def update_target(self, tau):
        for target_param, param in zip(self.actor_target.parameters(),
                                       self.actor_model.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1 - tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(),
                                       self.critic_model.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1 - tau) * target_param.data)
