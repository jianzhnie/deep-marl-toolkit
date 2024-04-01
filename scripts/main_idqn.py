"""This script contains the main function for running the IDQN (Independent
Deep Q-Network) algorithm in a StarCraft II environment."""

import argparse
import os
import sys
import time

import torch
from smac.env import StarCraft2Env
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../')
from configs.arguments import get_common_args
from configs.idqn_config import IDQNConfig
from marltoolkit.agents import IDQNAgent
from marltoolkit.data import MaReplayBuffer
from marltoolkit.envs.smacv1.env_wrapper import SC2EnvWrapper
from marltoolkit.modules.actors import RNNActor
from marltoolkit.runners.runner import run_evaluate_episode, run_train_episode
from marltoolkit.utils import (ProgressBar, TensorboardLogger, WandbLogger,
                               get_outdir, get_root_logger)


def main():
    """The main function for running the IDQN algorithm.

    It initializes the necessary components such as the environment, agent,
    logger, and replay buffer. Then, it performs training episodes and
    evaluation episodes, logging the results at specified intervals.
    """

    qmix_config = IDQNConfig()
    common_args = get_common_args()
    args = argparse.Namespace(**vars(common_args), **vars(qmix_config))
    device = (torch.device('cuda') if torch.cuda.is_available() and args.cuda
              else torch.device('cpu'))

    env = StarCraft2Env(map_name=args.scenario, difficulty=args.difficulty)

    env = SC2EnvWrapper(env)
    args.episode_limit = env.episode_limit
    args.obs_shape = env.obs_shape
    args.state_shape = env.state_shape
    args.num_agents = env.num_agents
    args.n_actions = env.n_actions
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

    rpm = MaReplayBuffer(
        buffer_size=args.replay_buffer_size,
        episode_limit=args.episode_limit,
        state_shape=args.state_shape,
        obs_shape=args.obs_shape,
        num_agents=args.num_agents,
        num_actions=args.n_actions,
        device=args.device,
    )

    agent_model = RNNActor(
        input_dim=args.obs_shape,
        rnn_hidden_dim=args.rnn_hidden_dim,
        n_actions=args.n_actions,
    )

    marl_agent = IDQNAgent(
        agent_model=agent_model,
        mixer_model=None,
        num_agents=args.num_agents,
        double_q=args.double_q,
        total_steps=args.total_steps,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        exploration_start=args.exploration_start,
        min_exploration=args.min_exploration,
        update_target_interval=args.update_target_interval,
        update_learner_freq=args.update_learner_freq,
        clip_grad_norm=args.clip_grad_norm,
        device=args.device,
    )

    progress_bar = ProgressBar(args.memory_warmup_size)
    while rpm.size() < args.memory_warmup_size:
        run_train_episode(env, marl_agent, rpm, args)
        progress_bar.update()

    steps_cnt = 0
    episode_cnt = 0
    progress_bar = ProgressBar(args.total_steps)
    while steps_cnt < args.total_steps:
        (
            episode_reward,
            episode_step,
            is_win,
            mean_loss,
            mean_td_error,
        ) = run_train_episode(env, marl_agent, rpm, args)
        # update episodes and steps
        episode_cnt += 1
        steps_cnt += episode_step

        # learning rate decay
        marl_agent.learning_rate = max(
            marl_agent.lr_scheduler.step(episode_step),
            marl_agent.min_learning_rate)

        train_results = {
            'env_step': episode_step,
            'rewards': episode_reward,
            'win_rate': is_win,
            'mean_loss': mean_loss,
            'mean_td_error': mean_td_error,
            'exploration': marl_agent.exploration,
            'learning_rate': marl_agent.learning_rate,
            'replay_buffer_size': rpm.size(),
            'target_update_count': marl_agent.target_update_count,
        }
        if episode_cnt % args.train_log_interval == 0:
            text_logger.info(
                '[Train], episode: {}, train_win_rate: {:.2f}, train_reward: {:.2f}'
                .format(episode_cnt, is_win, episode_reward))
            logger.log_train_data(train_results, steps_cnt)

        if episode_cnt % args.test_log_interval == 0:
            eval_rewards, eval_steps, eval_win_rate = run_evaluate_episode(
                env, marl_agent, num_eval_episodes=5)
            text_logger.info(
                '[Eval], episode: {}, eval_win_rate: {:.2f}, eval_rewards: {:.2f}'
                .format(episode_cnt, eval_win_rate, eval_rewards))

            test_results = {
                'env_step': eval_steps,
                'rewards': eval_rewards,
                'win_rate': eval_win_rate,
            }
            logger.log_test_data(test_results, steps_cnt)

        progress_bar.update(episode_step)


if __name__ == '__main__':
    main()
