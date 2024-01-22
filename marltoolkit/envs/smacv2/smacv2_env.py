import random

import numpy as np
from gymnasium.spaces import Discrete
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper


class SMACv2Env(StarCraftCapabilityEnvWrapper):

    def __init__(self, **kwargs):
        super(SMACv2Env, self).__init__(obs_last_action=False, **kwargs)
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        self.n_agents = self.env.n_agents

        for i in range(self.env.n_agents):
            self.action_space.append(Discrete(self.env.n_actions))
            self.observation_space.append([self.env.get_obs_size()])
            self.share_observation_space.append([self.env.get_state_size()])

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def step(self, actions):
        reward, terminated, info = self.env.step(actions)
        obs = self.env.get_obs()
        state = self.env.get_state()
        global_state = [state] * self.n_agents
        rewards = [[reward]] * self.n_agents
        dones = [terminated] * self.n_agents
        infos = [info] * self.n_agents
        avail_actions = self.env.get_avail_actions()

        bad_transition = True if self.env._episode_steps >= self.env.episode_limit else False
        for info in infos:
            info['bad_transition'] = bad_transition
            info['battles_won'] = self.env.battles_won
            info['battles_game'] = self.env.battles_game
            info['battles_draw'] = self.env.timeouts
            info['restarts'] = self.env.force_restarts
            info['won'] = self.env.win_counted

        return obs, global_state, rewards, dones, infos, avail_actions

    def reset(self):
        obs, state = super().reset()
        state = [state for i in range(self.env.n_agents)]
        avail_actions = self.env.get_avail_actions()
        return obs, state, avail_actions
