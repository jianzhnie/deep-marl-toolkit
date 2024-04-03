import sys

import torch
from torch.distributions import Categorical

sys.path.append('../')
from marltoolkit.envs.smacv1.smac_env import SMACWrapperEnv
from marltoolkit.envs.vec_env import DummyVecEnv, SubprocVecEnv


def make_envs(map_name='3m', parallels=8):

    def _thunk():
        env = SMACWrapperEnv(map_name=map_name)
        return env

    return SubprocVecEnv([_thunk for _ in range(parallels)],
                         shared_memory=False)


def make_dummy_envs(map_name='3m', parallels=8):

    def _thunk():
        env = SMACWrapperEnv(map_name=map_name)
        return env

    return DummyVecEnv([_thunk for _ in range(parallels)])


if __name__ == '__main__':
    train_envs = make_envs()
    results = train_envs.reset()
    print('Reset:', results)

    env_info = train_envs.get_env_info()
    print('env_info:', env_info)
    action_dim = env_info['n_actions']
    num_agents = env_info['num_agents']
    num_envs = 8

    avail_actions = train_envs.get_available_actions()

    print('avail_actions:', avail_actions)
    available_actions = torch.tensor(avail_actions)
    actions_dist = Categorical(available_actions)
    random_actions = actions_dist.sample().numpy()
    print('random_actions: ', random_actions)
    print('random_actions shape: ', random_actions.shape)

    results = train_envs.step(random_actions)
    print('Step:', results)
    train_envs.close()
