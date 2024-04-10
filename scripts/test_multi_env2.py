import sys

import torch
from torch.distributions import Categorical

sys.path.append('../')
from marltoolkit.envs.smacv1.smac_env import SMACWrapperEnv
from marltoolkit.envs.vec_env import DummyVecEnv, SubprocVecEnv


def make_subproc_envs(map_name='3m', parallels=8) -> SubprocVecEnv:

    def _thunk():
        env = SMACWrapperEnv(map_name=map_name)
        return env

    return SubprocVecEnv([_thunk for _ in range(parallels)],
                         shared_memory=False)


def make_dummy_envs(map_name='3m', parallels=8) -> DummyVecEnv:

    def _thunk():
        env = SMACWrapperEnv(map_name=map_name)
        return env

    return DummyVecEnv([_thunk for _ in range(parallels)])


def test_dummy_envs():
    parallels = 2
    env = SMACWrapperEnv(map_name='3m')
    env.reset()
    env_info = env.get_env_info()
    print('env_info:', env_info)

    train_envs = make_dummy_envs(parallels=parallels)
    results = train_envs.reset()
    print('Reset:', results)

    num_envs = parallels
    avail_actions = env.get_available_actions()
    print('avail_actions:', avail_actions)
    available_actions = torch.tensor(avail_actions)
    actions_dist = Categorical(available_actions)
    random_actions = actions_dist.sample().numpy().tolist()

    print('random_actions: ', random_actions)
    dummy_actions = [random_actions for _ in range(num_envs)]
    print('dummy_actions:', dummy_actions)
    results = train_envs.step(dummy_actions)
    print('Step:', results)
    train_envs.close()


def test_subproc_envs():
    parallels = 10
    train_envs = make_subproc_envs(parallels=parallels)
    results = train_envs.reset()
    obs, state, info = results
    print('Env Reset:', '*' * 100)
    print('obs:', obs.shape)
    print('state:', state.shape)
    print('info:', info)

    avail_actions = train_envs.get_available_actions()
    print('avail_actions:', avail_actions)
    available_actions = torch.tensor(avail_actions)
    actions_dist = Categorical(available_actions)
    random_actions = actions_dist.sample().numpy().tolist()
    print('random_actions: ', random_actions)
    results = train_envs.step(random_actions)
    obs, state, rewards, dones, info = results
    print('Env Step:', '*' * 100)
    print('obs:', obs.shape)
    print('state:', state.shape)
    print('rewards:', rewards)
    print('dones:', dones)
    print('info:', info)


if __name__ == '__main__':
    test_subproc_envs()
