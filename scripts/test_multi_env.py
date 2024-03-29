import sys

sys.path.append('../')
from marltoolkit.envs.smacv1 import SMACWrapperEnv, SubprocVecSMAC


def make_envs(map_name='3m', parallels=8):

    def _thunk():
        env = SMACWrapperEnv(map_name=map_name)
        return env

    return SubprocVecSMAC([_thunk for _ in range(parallels)])


if __name__ == '__main__':
    train_envs = make_envs()
    train_envs.reset()
    actions = train_envs.get_avail_actions()
    print(actions)
    train_envs.close()
