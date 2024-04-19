import argparse


def get_common_args() -> argparse.Namespace:
    """The configuration parser for common hyperparameters of all environment.
    Please reach each `scripts/train/<env>_runner.py` file to find private
    hyperparameters only used in <env>.

    Prepare parameters:
        --algorithm_name <algorithm_name>
            specifiy the algorithm, including `["rmappo", "mappo", "rmappg", "mappg", "trpo"]`
        --seed <int>
            set seed for numpy and torch
        --cuda
            by default True, will use GPU to train; or else will use CPU;
        --cuda_deterministic
            by default, make sure random seed effective. if set, bypass such function.
        --n_training_threads <int>
            number of training threads working in parallel. by default 1
        --n_eval_threads <int>
            number of evaluation threads working in parallel. by default 1
        --total_steps <int>
            number of env steps to train (default: 10e6)
        --use_wandb
            [for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.

    Env parameters:
        --project <str>
            The Project name
        --env_name <str>
            specify the name of environment
        --map_name <str>
            specify the map_name of environment
        --difficulty <str>
            specify the difficulty of environment
        --delta_time <float>
            delta time per step
        --step_mul <int>
            how many steps to make an action
        --num_train_envs <int>
            number of parallel envs for training rollout. by default 2
        --num_test_envs <int>
            number of parallel envs for evaluating rollout. by default 1
        --use_global_state <bool>
            whether to use the global state
        --use_last_actions <bool>
            whether to use the last action
        --use_agents_id_onehot <bool>
            whether to use the agent id transform

    Network parameters:
        --share_policy
            by default True, all agents will share the same network; set to make training agents use different policies.
        --use_centralized_v
            by default True, use centralized training mode; or else will decentralized training mode.
        --stacked_frames <int>
            Number of input frames which should be stack together.
        --hidden_size <int>
            Dimension of hidden layers for actor/critic networks
        --use_relu
            by default True, will use ReLU. or else will use Tanh.
        --use_popart
            by default True, use PopArt to normalize rewards.
        --use_valuenorm
            by default True, use running mean and std to normalize rewards.
        --use_feature_normalization
            by default True, apply layernorm to normalize inputs.
        --use_orthogonal
            by default True, use Orthogonal initialization for weights and 0 initialization for biases. or else, will use xavier uniform inilialization.
        --gain
            by default 0.01, use the gain # of last action layer
        --use_naive_recurrent_policy
            by default False, use the whole trajectory to calculate hidden states.
        --use_recurrent_policy
            by default, use Recurrent Policy. If set, do not use.
        --rnn_layers <int>
            The number of recurrent layers ( default 1).
        --data_chunk_length <int>
            Time length of chunks used to train a recurrent_policy, default 10.

    PPO parameters:
        --ppo_epoch <int>
            number of ppo epochs (default: 15)
        --use_clipped_value_loss
            by default, clip loss value. If set, do not clip loss value.
        --clip_param <float>
            ppo clip parameter (default: 0.2)
        --num_mini_batch <int>
            number of batches for ppo (default: 1)
        --entropy_coef <float>
            entropy term coefficient (default: 0.01)
        --use_max_grad_norm
            by default, use max norm of gradients. If set, do not use.
        --max_grad_norm <float>
            max norm of gradients (default: 0.5)
        --use_gae
            by default, use generalized advantage estimation. If set, do not use gae.
        --gamma <float>
            discount factor for rewards (default: 0.99)
        --gae_lambda <float>
            gae lambda parameter (default: 0.95)
        --use_proper_time_limits
            by default, the return value does consider limits of time. If set, compute returns with considering time limits factor.
        --use_huber_loss
            by default, use huber loss. If set, do not use huber loss.
        --use_value_active_masks
            by default True, whether to mask useless data in value loss.
        --huber_delta <float>
            coefficient of huber loss.

    PPG parameters:
        --aux_epoch <int>
            number of auxiliary epochs. (default: 4)
        --clone_coef <float>
            clone term coefficient (default: 0.01)

    Replay Buffer parameters:
        --replay_buffer_size <int>
            max number of episodes stored in the replay buffer.
        --memory_warmup_size <int>
            Learning start until replay_buffer_size >= 'memory_warmup_size'
        --episode_length <int>
            the max length of episode in the buffer.

    Optimizer parameters:
        --learning_rate <float>
            learning rate parameter,  (default: 5e-4, fixed).
        --min_learning_rate <float>
            min learning rate parameter, (default: 1e-6, fixed).
        --actor_lr <float>
            learning rate of actor  (default: 5e-4, fixed)
        --critic_lr <float>
            learning rate of critic  (default: 5e-4, fixed)
        --opti_eps <float>
            RMSprop optimizer epsilon (default: 1e-5)
        --weight_decay <float>
            coefficience of weight decay (default: 0)

    Train and test parameters:
        --use_linear_lr_decay
            by default, do not apply linear decay to learning rate. If set, use a linear schedule on the learning rate
        --batch_size <int>
            Training batch size.
        --test_interval <int>
            Evaluate the model every 'test_steps' steps.
        --num_test_episodes <int>
            number of episodes of a single evaluation.

    Save & Log parameters:
        --log_dir <str>
            the directory to save the log and model
        --train_log_interval <int>
            Log interval(Eposide) for training
        --test_log_interval <int>
            Log interval(Eposide) for testing.
        --save_interval <int>
            time duration between contiunous twice models saving.

    Render parameters:
        --save_gifs
            by default, do not save render video. If set, save video.
        --use_render
            by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.
        --render_episodes <int>
            the number of episodes to render a given env
        --ifi <float>
            the play interval of each rendered image in saved video.


    Pretrained parameters:
        --load_model <str>
            whether to load the pretrained model
        --model_dir <str>
            by default None. set the path to pretrained model.
    """

    parser = argparse.ArgumentParser()

    # prepare parameters
    parser.add_argument(
        '--algorithm_name',
        type=str,
        default='rmappo',
        choices=['rmappo', 'mappo', 'happo', 'hatrpo', 'mat', 'mat_dec'],
    )
    parser.add_argument('--seed',
                        type=int,
                        default=123,
                        help='random seed for numpy/torch')
    parser.add_argument(
        '--cuda',
        action='store_false',
        default=True,
        help='by default True, will use GPU to train; or else will use CPU;',
    )
    parser.add_argument(
        '--cuda_deterministic',
        action='store_false',
        default=True,
        help=
        'by default, make sure random seed effective. if set, bypass such function.',
    )
    parser.add_argument(
        '--n_training_threads',
        type=int,
        default=1,
        help='Number of torch threads for training',
    )
    parser.add_argument(
        '--n_eval_threads',
        type=int,
        default=1,
        help='Number of torch threads for  evaluating',
    )
    parser.add_argument(
        '--total_steps',
        type=int,
        default=10e6,
        help='Number of environment steps to train (default: 10e6)',
    )

    # The Environment setting
    parser.add_argument(
        '--project',
        type=str,
        default='StarCraft2',
        help='The project which env is in',
    )
    parser.add_argument('--env_name',
                        type=str,
                        default='SMAC-v1',
                        help='the map of the game')
    parser.add_argument('--map_name',
                        type=str,
                        default='3m',
                        help='the map of the game')
    parser.add_argument('--difficulty',
                        type=str,
                        default='7',
                        help='Difficulty of the environment.')
    parser.add_argument('--delta_time',
                        type=float,
                        default=1,
                        help='delta time per step')
    parser.add_argument('--step_mul',
                        type=int,
                        default=8,
                        help='how many steps to make an action')
    parser.add_argument(
        '--num_train_envs',
        type=int,
        default=1,
        help='Num of parallel threads running the env',
    )
    parser.add_argument(
        '--num_test_envs',
        type=int,
        default=1,
        help='Num of parallel threads running the env',
    )
    parser.add_argument(
        '--use_global_state',
        type=bool,
        default=False,
        help='whether to use the global state',
    )
    parser.add_argument(
        '--use_last_actions',
        type=bool,
        default=True,
        help='whether to use the last action',
    )
    parser.add_argument(
        '--use_agents_id_onehot',
        type=bool,
        default=True,
        help='whether to use the agent id transform',
    )

    # network parameters
    parser.add_argument(
        '--share_policy',
        action='store_false',
        default=True,
        help='Whether agent share the same policy',
    )
    parser.add_argument(
        '--use_centralized_v',
        action='store_false',
        default=True,
        help='Whether to use centralized V function',
    )
    parser.add_argument(
        '--stacked_frames',
        type=int,
        default=1,
        help='Dimension of hidden layers for actor/critic networks',
    )
    parser.add_argument(
        '--use_stacked_frames',
        action='store_true',
        default=False,
        help='Whether to use stacked_frames',
    )
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=64,
        help='Dimension of hidden layers for actor/critic networks',
    )
    parser.add_argument('--activation',
                        default='relu',
                        help='Whether to use ReLU')
    parser.add_argument(
        '--use_popart',
        action='store_true',
        default=False,
        help='by default False, use PopArt to normalize rewards.',
    )
    parser.add_argument(
        '--use_valuenorm',
        action='store_false',
        default=True,
        help='by default True, use running mean and std to normalize rewards.',
    )
    parser.add_argument(
        '--use_feature_normalization',
        action='store_false',
        default=True,
        help='Whether to apply layernorm to the inputs',
    )
    parser.add_argument(
        '--use_orthogonal',
        action='store_false',
        default=True,
        help=
        'Whether to use Orthogonal initialization for weights and 0 initialization for biases',
    )
    parser.add_argument('--gain',
                        type=float,
                        default=0.01,
                        help='The gain # of last action layer')

    # recurrent parameters
    parser.add_argument(
        '--use_naive_recurrent_policy',
        action='store_true',
        default=False,
        help='Whether to use a naive recurrent policy',
    )
    parser.add_argument(
        '--use_recurrent_policy',
        action='store_false',
        default=True,
        help='use a recurrent policy',
    )
    parser.add_argument('--rnn_layers',
                        type=int,
                        default=2,
                        help='The number of recurrent layers.')
    parser.add_argument(
        '--data_chunk_length',
        type=int,
        default=10,
        help='Time length of chunks used to train a recurrent_policy',
    )
    # trpo parameters
    parser.add_argument(
        '--kl_threshold',
        type=float,
        default=0.01,
        help='the threshold of kl-divergence (default: 0.01)',
    )
    parser.add_argument('--ls_step',
                        type=int,
                        default=10,
                        help='number of line search (default: 10)')
    parser.add_argument(
        '--accept_ratio',
        type=float,
        default=0.5,
        help='accept ratio of loss improve (default: 0.5)',
    )

    # ppo parameters
    parser.add_argument('--ppo_epoch',
                        type=int,
                        default=15,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument(
        '--use_clipped_value_loss',
        action='store_false',
        default=True,
        help='by default, clip loss value. If set, do not clip loss value.',
    )
    parser.add_argument(
        '--clip_param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)',
    )
    parser.add_argument(
        '--num_mini_batch',
        type=int,
        default=1,
        help='number of batches for ppo (default: 1)',
    )
    parser.add_argument(
        '--entropy_coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)',
    )
    parser.add_argument(
        '--value_loss_coef',
        type=float,
        default=1,
        help='value loss coefficient (default: 0.5)',
    )
    parser.add_argument(
        '--use_max_grad_norm',
        action='store_false',
        default=True,
        help='by default, use max norm of gradients. If set, do not use.',
    )
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=10.0,
        help='max norm of gradients (default: 0.5)',
    )
    parser.add_argument(
        '--use_gae',
        action='store_false',
        default=True,
        help='use generalized advantage estimation',
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)',
    )
    parser.add_argument(
        '--gae_lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)',
    )
    parser.add_argument(
        '--use_proper_time_limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits',
    )
    parser.add_argument(
        '--use_huber_loss',
        action='store_false',
        default=True,
        help='by default, use huber loss. If set, do not use huber loss.',
    )
    parser.add_argument(
        '--use_value_active_masks',
        action='store_false',
        default=True,
        help='by default True, whether to mask useless data in value loss.',
    )
    parser.add_argument(
        '--use_policy_active_masks',
        action='store_false',
        default=True,
        help='by default True, whether to mask useless data in policy loss.',
    )
    parser.add_argument('--huber_delta',
                        type=float,
                        default=10.0,
                        help=' coefficience of huber loss.')

    # Replay buffer parameters
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

    # Optimizer parameters
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.001,
                        help='Learning rate.')
    parser.add_argument('--min_learning_rate',
                        type=float,
                        default=1e-6,
                        help='Min learning rate.')
    parser.add_argument(
        '--actor_lr',
        type=float,
        default=5e-4,
        help='actor learning rate (default: 5e-4)',
    )
    parser.add_argument(
        '--critic_lr',
        type=float,
        default=5e-4,
        help='critic learning rate (default: 5e-4)',
    )
    parser.add_argument(
        '--opti_eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)',
    )
    parser.add_argument('--weight_decay', type=float, default=0)

    # Train Parameters
    parser.add_argument(
        '--target_update_tau',
        type=float,
        default=0.05,
        help='Target model soft update coef.',
    )
    parser.add_argument(
        '--target_update_interval',
        type=int,
        default=100,
        help='Target model update interval.',
    )
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Training batch size.')
    parser.add_argument(
        '--test_interval',
        type=int,
        default=100,
        help="Evaluate the model every 'test_steps' steps.",
    )
    parser.add_argument(
        '--num_test_episodes',
        type=int,
        default=32,
        help='number of episodes of a single evaluation.',
    )

    # log parameters
    parser.add_argument(
        '--logger',
        type=str,
        default='wandb',
        help=
        '[for wandb usage], by default use wandb logger, will log date to wandb server. or else will use tensorboard to log data.',
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./work_dirs',
        help='the directory to save the log and model',
    )
    parser.add_argument(
        '--train_log_interval',
        type=int,
        default=5,
        help='Log interval(Eposide) for training',
    )
    parser.add_argument(
        '--test_log_interval',
        type=int,
        default=20,
        help='Log interval(Eposide) for testing.',
    )

    # Render parameters
    parser.add_argument(
        '--save_gifs',
        action='store_true',
        default=False,
        help='by default, do not save render video. If set, save video.',
    )
    parser.add_argument(
        '--use_render',
        action='store_true',
        default=False,
        help=
        'by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.',
    )
    parser.add_argument(
        '--render_episodes',
        type=int,
        default=5,
        help='the number of episodes to render a given env',
    )
    parser.add_argument(
        '--ifi',
        type=float,
        default=0.1,
        help='the play interval of each rendered image in saved video.',
    )

    # Pretrained parameters
    parser.add_argument(
        '--load_model',
        type=bool,
        default=False,
        help='whether to load the pretrained model',
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default=None,
        help='by default None. set the path to pretrained model.',
    )
    args = parser.parse_args()
    return args
