import sys
from argparse import Namespace

sys.path.append('../')
from configs.arguments import get_common_args
from marltoolkit.envs.smac import SC2Env, SubprocVecEnvSC2


def make_envs(args: Namespace):

    def _thunk():
        env = SC2Env(map_name=args.scenario)
        return env

    return SubprocVecEnvSC2([_thunk for _ in range(args.parallels)])


if __name__ == '__main__':
    common_args = get_common_args()
    envs = make_envs(common_args)
    envs.reset()
    actions = envs.get_available_actions()
    print(actions)
    print(actions.shape)
