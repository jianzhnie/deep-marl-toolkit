import os
import sys
import time

import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../')

from configs.arguments import get_common_args
from marltoolkit.agents.mappo_agent import MAPPOAgent
from marltoolkit.data.shared_buffer import SharedReplayBuffer
from marltoolkit.envs.smacv1 import SMACWrapperEnv
from marltoolkit.runners.episode_runner import (run_eval_episode,
                                                run_train_episode)
from marltoolkit.utils import (ProgressBar, TensorboardLogger, WandbLogger,
                               get_outdir, get_root_logger)


def main() -> None:
    """Main function for running the QMix algorithm.

    This function initializes the necessary configurations, environment, logger, models, and agents.
    It then runs training episodes and evaluates the agent's performance periodically.

    Returns:
        None
    """
    args = get_common_args()
    if args.algorithm_name == 'rmappo':
        args.use_recurrent_policy = True
        args.use_naive_recurrent_policy = False

    if args.algorithm_name == 'rmappo':
        print(
            'u are choosing to use rmappo, we set use_recurrent_policy to be True'
        )
        args.use_recurrent_policy = True
        args.use_naive_recurrent_policy = False
    elif (args.algorithm_name == 'mappo' or args.algorithm_name == 'mat'
          or args.algorithm_name == 'mat_dec'):
        assert (args.use_recurrent_policy is False
                and args.use_naive_recurrent_policy is False
                ), 'check recurrent policy!'
        print(
            'U are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False'
        )
        args.use_recurrent_policy = False
        args.use_naive_recurrent_policy = False

    elif args.algorithm_name == 'ippo':
        print(
            'u are choosing to use ippo, we set use_centralized_v to be False')
        args.use_centralized_v = False
    elif args.algorithm_name == 'happo' or args.algorithm_name == 'hatrpo':
        # can or cannot use recurrent network?
        print('using', args.algorithm_name, 'without recurrent network')
        args.use_recurrent_policy = False
        args.use_naive_recurrent_policy = False
    else:
        raise NotImplementedError

    if args.algorithm_name == 'mat_dec':
        args.dec_actor = True
        args.share_actor = True

    # cuda
    if args.cuda and torch.cuda.is_available():
        print('choose to use gpu...')
        args.device = torch.device('cuda')
        torch.set_num_threads(args.n_training_threads)
        if args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print('choose to use cpu...')
        args.device = torch.device('cpu')
        torch.set_num_threads(args.n_training_threads)

    # Environment
    env = SMACWrapperEnv(map_name=args.scenario, args=args)
    args.episode_limit = env.episode_limit
    args.obs_dim = env.obs_dim
    args.obs_shape = env.obs_shape
    args.state_dim = env.state_dim
    args.state_shape = env.state_shape
    args.num_agents = env.num_agents
    args.n_actions = env.n_actions
    args.action_shape = env.action_shape
    args.reward_shape = env.reward_shape
    args.done_shape = env.done_shape

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # log
    log_name = os.path.join(args.project, args.scenario, args.algo_name,
                            timestamp).replace(os.path.sep, '_')
    log_path = os.path.join(args.log_dir, args.project, args.scenario,
                            args.algo_name)
    tensorboard_log_path = get_outdir(log_path, 'tensorboard_log_dir')
    log_file = os.path.join(log_path, log_name + '.log')
    text_logger = get_root_logger(log_file=log_file, log_level='INFO')

    if args.logger == 'wandb':
        logger = WandbLogger(
            train_interval=args.train_log_interval,
            test_interval=args.test_log_interval,
            update_interval=args.train_log_interval,
            project=args.project,
            name=log_name,
            save_interval=1,
            config=args,
        )
    writer = SummaryWriter(tensorboard_log_path)
    writer.add_text('args', str(args))
    if args.logger == 'tensorboard':
        logger = TensorboardLogger(writer)
    else:  # wandb
        logger.load(writer)

    args.obs_shape = env.get_actor_input_shape()

    # policy network
    agent = MAPPOAgent(args)
    # buffer
    rpm = SharedReplayBuffer(
        args.num_envs,
        args.num_agents,
        args.episode_limit,
        args.obs_shape,
        args.state_shape,
        args.action_shape,
        args.reward_shape,
        args.done_shape,
        args,
    )

    progress_bar = ProgressBar(args.memory_warmup_size)
    while rpm.size() < args.memory_warmup_size:
        run_train_episode(env, agent, rpm, args)
        progress_bar.update()

    steps_cnt = 0
    episode_cnt = 0
    progress_bar = ProgressBar(args.total_steps)
    while steps_cnt < args.total_steps:
        train_res_dict = run_train_episode(env, agent, rpm, args)
        # update episodes and steps
        episode_cnt += 1
        steps_cnt += train_res_dict['episode_step']

        # learning rate decay
        agent.learning_rate = max(
            agent.lr_scheduler.step(train_res_dict['episode_step']),
            agent.min_learning_rate,
        )

        train_res_dict.update({
            'exploration': agent.exploration,
            'learning_rate': agent.learning_rate,
            'replay_max_size': rpm.size(),
            'target_update_count': agent.target_update_count,
        })
        if episode_cnt % args.train_log_interval == 0:
            text_logger.info(
                '[Train], episode: {}, train_episode_step: {}, train_win_rate: {:.2f}, train_reward: {:.2f}'
                .format(
                    episode_cnt,
                    train_res_dict['episode_step'],
                    train_res_dict['win_rate'],
                    train_res_dict['episode_reward'],
                ))
            logger.log_train_data(train_res_dict, steps_cnt)

        if episode_cnt % args.test_log_interval == 0:
            eval_res_dict = run_eval_episode(env, agent, args=args)
            text_logger.info(
                '[Eval], episode: {}, eval_episode_step:{:.2f}, eval_win_rate: {:.2f}, eval_reward: {:.2f}'
                .format(
                    episode_cnt,
                    eval_res_dict['episode_step'],
                    eval_res_dict['win_rate'],
                    eval_res_dict['episode_reward'],
                ))
            logger.log_test_data(eval_res_dict, steps_cnt)

        progress_bar.update(train_res_dict['episode_step'])


if __name__ == '__main__':
    main()
