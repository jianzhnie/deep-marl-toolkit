import torch


class MaddpgAgent(object):

    def __init__(self) -> None:
        super().__init__()
        self.actor = None
        self.critic = None
        self.actor_target = None
        self.critic_target = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.n_actions = None
        self.n_agents = None
        self.input_shape = None
        self.hidden_dim = None
        self.device = None

    def sample_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def predict_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def update_target(self, tau):
        for target_param, param in zip(self.actor_target.parameters(),
                                       self.actor.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1 - tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(),
                                       self.critic.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1 - tau) * target_param.data)

    def learn(self, batch, gamma, batch_size):
        raise NotImplementedError
