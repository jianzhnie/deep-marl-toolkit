import sys

sys.path.append('../')
from marltoolkit.envs.smacv1.smac_env import SMACWrapperEnv
from marltoolkit.envs.vec_env import SubprocVecEnv


def make_envs(map_name='3m', parallels=8):

    def _thunk():
        env = SMACWrapperEnv(map_name=map_name)
        return env

    return SubprocVecEnv([_thunk for _ in range(parallels)])


if __name__ == '__main__':
    train_envs = make_envs()
    results = train_envs.reset()
    print(results)
    train_envs.close()
