from abc import ABC
from typing import Any, Dict, List, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils.wrappers import BaseWrapper


class PettingZooEnv(AECEnv, ABC):

    def __init__(self, env: BaseWrapper):
        super(PettingZooEnv, self).__init__()
        self.env = env
        self.agents = self.env.possible_agents
        self.agent_idx = {}
        for i, agent_id in enumerate(self.agents):
            self.agent_idx[agent_id] = i

        self.action_spaces = {
            k: self.env.action_space(k)
            for k in self.env.agents
        }
        self.observation_spaces = {
            k: self.env.observation_space(k)
            for k in self.env.agents
        }
        assert all(
            self.observation_space(agent) == self.env.observation_space(
                self.agents[0])
            for agent in self.agents), (
                'Observation spaces for all agents must be identical. Perhaps '
                "SuperSuit's pad_observations wrapper can help (useage: "
                '`supersuit.aec_wrappers.pad_observations(env)`')

        assert all(
            self.action_space(agent) == self.env.action_space(self.agents[0])
            for agent in self.agents), (
                'Action spaces for all agents must be identical. Perhaps '
                "SuperSuit's pad_action_space wrapper can help (useage: "
                '`supersuit.aec_wrappers.pad_action_space(env)`')
        try:
            self.state_space = self.env.state_space
        except Exception:
            self.state_space = None

        self.rewards = [0] * len(self.agents)
        self.metadata = self.env.metadata
        self.env.reset()

    def reset(self, *args: Any, **kwargs: Any) -> Tuple[dict, dict]:
        self.env.reset(*args, **kwargs)

        observation, reward, terminated, truncated, info = self.env.last(self)

        if isinstance(observation, dict) and 'action_mask' in observation:
            observation_dict = {
                'agent_id':
                self.env.agent_selection,
                'obs':
                observation['observation'],
                'mask': [
                    True if obm == 1 else False
                    for obm in observation['action_mask']
                ],
            }
        else:
            if isinstance(self.action_space, spaces.Discrete):
                observation_dict = {
                    'agent_id':
                    self.env.agent_selection,
                    'obs':
                    observation,
                    'mask':
                    [True] * self.env.action_space(self.env.agent_selection).n,
                }
            else:
                observation_dict = {
                    'agent_id': self.env.agent_selection,
                    'obs': observation,
                }

        return observation_dict, info

    def step(self, action: Any) -> Tuple[Dict, List[int], bool, bool, Dict]:
        self.env.step(action)

        observation, reward, term, trunc, info = self.env.last()

        if isinstance(observation, dict) and 'action_mask' in observation:
            obs = {
                'agent_id':
                self.env.agent_selection,
                'obs':
                observation['observation'],
                'mask': [
                    True if obm == 1 else False
                    for obm in observation['action_mask']
                ],
            }
        else:
            if isinstance(self.action_space, spaces.Discrete):
                obs = {
                    'agent_id':
                    self.env.agent_selection,
                    'obs':
                    observation,
                    'mask':
                    [True] * self.env.action_space(self.env.agent_selection).n,
                }
            else:
                obs = {
                    'agent_id': self.env.agent_selection,
                    'obs': observation
                }

        for agent_id, reward in self.env.rewards.items():
            self.rewards[self.agent_idx[agent_id]] = reward
        return obs, self.rewards, term, trunc, info

    def state(self):
        try:
            return np.array(self.env.state())
        except Exception:
            return None

    def seed(self, seed: Any = None) -> None:
        try:
            self.env.seed(seed)
        except (NotImplementedError, AttributeError):
            self.env.reset(seed=seed)

    def render(self) -> Any:
        return self.env.render()

    def close(self):
        self.env.close()
