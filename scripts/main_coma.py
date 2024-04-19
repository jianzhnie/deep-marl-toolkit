import argparse
import os
import sys
import time

import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../')

from configs.arguments import get_common_args
from configs.coma_config import ComaConfig
from marltoolkit.agents.coma_agent import ComaAgent
from marltoolkit.data import ReplayBuffer
from marltoolkit.envs.smacv1.smac_env import SMACWrapperEnv
from marltoolkit.modules.actors import RNNActorModel
from marltoolkit.modules.critics.coma import MLPCriticModel
from marltoolkit.runners.episode_runner import (run_eval_episode,
                                                run_train_episode)
from marltoolkit.utils import (ProgressBar, TensorboardLogger, WandbLogger,
                               get_outdir, get_root_logger)


def main():
    """Main function for running the QMix algorithm.

    This function initializes the necessary configurations, environment, logger, models, and agents.
    It then runs training episodes and evaluates the agent's performance periodically.

    Returns:
        None
    """
    coma_config = ComaConfig()
    common_args = get_common_args()
    args = argparse.Namespace(**vars(common_args), **vars(coma_config))
    device = (torch.device('cuda') if torch.cuda.is_available() and args.cuda
              else torch.device('cpu'))

    env = SMACWrapperEnv(map_name=args.scenario, difficulty=args.difficulty)
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
    args.device = device

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

    rpm = ReplayBuffer(
        max_size=args.replay_buffer_size,
        num_agents=args.num_agents,
        episode_limit=args.episode_limit,
        obs_space=args.obs_shape,
        state_space=args.state_shape,
        action_space=args.action_shape,
        reward_space=args.reward_shape,
        done_space=args.done_shape,
        device=device,
    )

    actor_model = RNNActorModel(
        input_dim=args.obs_dim,
        rnn_hidden_dim=args.rnn_hidden_dim,
        n_actions=args.n_actions,
    )

    critic_model = MLPCriticModel(
        input_dim=args.obs_dim,
        hidden_dim=args.fc_hidden_dim,
        output_dim=args.n_actions,
    )
    agent = ComaAgent(
        actor_model=actor_model,
        critic_model=critic_model,
        num_agents=args.num_agents,
        double_q=args.double_q,
        total_steps=args.total_steps,
        gamma=args.gamma,
        actor_lr=args.learning_rate,
        critic_lr=args.learning_rate,
        egreedy_exploration=args.egreedy_exploration,
        min_exploration=args.min_exploration,
        target_update_interval=args.target_update_interval,
        learner_update_freq=args.learner_update_freq,
        clip_grad_norm=args.clip_grad_norm,
        device=args.device,
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
