import argparse


def get_common_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument(
        '--project',
        type=str,
        default='StarCraft2',
        help='The project which env is in',
    )
    parser.add_argument('--env_id',
                        type=str,
                        default='SMAC-v1',
                        help='the map of the game')
    parser.add_argument('--scenario',
                        type=str,
                        default='3m',
                        help='the map of the game')
    parser.add_argument('--difficulty',
                        type=str,
                        default='7',
                        help='Difficulty of the environment.')
    parser.add_argument(
        '--num_train_envs',
        type=int,
        default=2,
        help='Num of parallel threads running the env',
    )
    parser.add_argument(
        '--num_test_envs',
        type=int,
        default=1,
        help='Num of parallel threads running the env',
    )
    parser.add_argument(
        '--use_gloabl_state',
        type=bool,
        default=False,
        help='whether to use the global state',
    )
    parser.add_argument(
        '--use_last_action',
        type=bool,
        default=True,
        help='whether to use the last action',
    )
    parser.add_argument(
        '--use_agent_id_onehot',
        type=bool,
        default=True,
        help='whether to use the agent id transform',
    )
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--total_steps',
                        type=int,
                        default=1000000,
                        help='total episode')
    parser.add_argument(
        '--replay_buffer_size',
        type=int,
        default=5000,
        help='Max number of episodes stored in the replay buffer.',
    )
    parser.add_argument(
        '--memory_warmup_size',
        type=int,
        default=32,
        help="Learning start until replay_buffer_size >= 'memory_warmup_size'",
    )
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Training batch size.')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./work_dirs',
        help='result directory of the policy',
    )
    parser.add_argument('--logger',
                        type=str,
                        default='wandb',
                        help='the logger for the experiment')
    parser.add_argument(
        '--train_log_interval',
        type=int,
        default=5,
        help='Log interval(Eposide) for training',
    )
    parser.add_argument('--test_log_interval',
                        type=int,
                        default=20,
                        help='Log interval for testing.')
    parser.add_argument(
        '--test_steps',
        type=int,
        default=100,
        help="Evaluate the model every 'test_steps' steps.",
    )
    parser.add_argument(
        '--load_model',
        type=bool,
        default=False,
        help='whether to load the pretrained model',
    )
    parser.add_argument('--stats',
                        type=str,
                        default='',
                        help='the stats file for data normalization')
    parser.add_argument('--delta_time',
                        type=float,
                        default=1,
                        help='delta time per step')
    parser.add_argument('--step_mul',
                        type=int,
                        default=8,
                        help='how many steps to make an action')
    parser.add_argument('--cuda',
                        type=bool,
                        default=True,
                        help='whether to use the GPU')
    args = parser.parse_args()
    return args
