import argparse


def get_common_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument(
        '--project',
        type=str,
        default='StarCraft-II',
        help='The project which env is in')
    parser.add_argument(
        '--scenario', type=str, default='3m', help='the map of the game')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument(
        '--total_steps', type=int, default=1000000, help='total episode')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./work_dirs',
        help='result directory of the policy',
    )
    parser.add_argument(
        '--logger',
        type=str,
        default='wandb',
        help='the logger for the experiment')
    parser.add_argument(
        '--load_model',
        type=bool,
        default=False,
        help='whether to load the pretrained model',
    )
    parser.add_argument(
        '--stats',
        type=str,
        default='',
        help='the stats file for data normalization')
    parser.add_argument(
        '--delta_time', type=float, default=1, help='delta time per step')
    parser.add_argument(
        '--step_mul',
        type=int,
        default=8,
        help='how many steps to make an action')
    parser.add_argument(
        '--cuda', type=bool, default=True, help='whether to use the GPU')
    args = parser.parse_args()
    return args
