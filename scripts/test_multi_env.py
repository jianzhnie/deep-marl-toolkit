import sys

sys.path.append('../')
from tianshou.env import ShmemVectorEnv

from configs.arguments import get_common_args
from marltoolkit.envs import SMACEnv, SubprocVecSMAC


def make_subproc_env(map_name: str = '3m', parallels: int = 10):
    """Wrapper function for Atari env.

    If EnvPool is installed, it will automatically switch to EnvPool's Atari env.

    :return: a tuple of (single env, training envs, test envs).
    """

    def _thunk(map_name):
        env = SMACEnv(map_name=map_name)
        return env

    train_envs = ShmemVectorEnv(
        [lambda: _thunk(map_name) for _ in range(parallels)])
    test_envs = ShmemVectorEnv(
        [lambda: _thunk(map_name) for _ in range(parallels)])
    return train_envs, test_envs


def make_envs(map_name: str = '3m', parallels: int = 10):

    def _thunk():
        print('*' * 50)
        print(map_name)
        env = SMACEnv(map_name=map_name)
        return env

    if parallels == 1:
        return _thunk()
    else:
        return SubprocVecSMAC([_thunk for _ in range(parallels)])


if __name__ == '__main__':
    common_args = get_common_args()
    env = SMACEnv(map_name='2m_vs_1z')
    env.reset()
    envs = make_subproc_env(map_name='3m', parallels=10)
    envs.reset()

    print('envs reset')
    print('==' * 50)
