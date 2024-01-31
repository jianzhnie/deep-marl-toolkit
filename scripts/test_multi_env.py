import sys

sys.path.append('../')
from configs.arguments import get_common_args
from marltoolkit.envs import SMACEnv, SubprocVecSMAC


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
    envs = make_envs(map_name='3m', parallels=1)
    envs.reset()

    print('envs reset')
    print('==' * 50)
    # envs = make_train_env()
    # print(envs)
    # envs.reset()
