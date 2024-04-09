import argparse

import numpy as np

from marltoolkit.agents import BaseAgent
from marltoolkit.data.ma_buffer import EpisodeData, ReplayBuffer
from marltoolkit.envs import MultiAgentEnv


def run_train_episode(
    env: MultiAgentEnv,
    agent: BaseAgent,
    rpm: ReplayBuffer,
    args: argparse.Namespace = None,
):
    episode_limit = args.episode_limit
    agent.init_hidden_states()
    episode_reward = 0.0
    episode_step = 0
    terminated = False
    (obs, state) = env.reset()
    episode_transition = EpisodeData(
        num_agents=args.num_agents,
        num_actions=args.n_actions,
        episode_limit=episode_limit,
        obs_shape=args.obs_shape,
        state_shape=args.state_shape,
    )

    while not terminated:
        available_actions = env.get_available_actions()
        actions = agent.sample(obs, available_actions)
        actions_onehot = env._get_actions_one_hot(actions)
        next_obs, next_state, reward, terminated, info = env.step(actions)
        episode_reward += reward
        episode_step += 1
        episode_transition.store(
            obs,
            state,
            actions,
            actions_onehot,
            available_actions,
            reward,
            terminated,
            filled=0,
        )
        state = next_state
        obs = next_obs

    # fill the episode
    for _ in range(episode_step, episode_limit):
        episode_transition.fill_mask()

    # get the whole episode data
    episode_data = episode_transition.get_episodes_data()

    # ReplayBuffer store the episode data
    rpm.store_episodes(**episode_data)
    is_win = env.win_counted

    mean_loss = []
    mean_td_error = []
    if rpm.size() > args.memory_warmup_size:
        for _ in range(args.update_learner_freq):
            batch = rpm.sample(args.batch_size)
            loss, td_error = agent.learn(**batch)
            mean_loss.append(loss)
            mean_td_error.append(td_error)

    mean_loss = np.mean(mean_loss) if mean_loss else None
    mean_td_error = np.mean(mean_td_error) if mean_td_error else None

    return episode_reward, episode_step, is_win, mean_loss, mean_td_error


def run_evaluate_episode(
    env: MultiAgentEnv,
    agent: BaseAgent,
    num_eval_episodes: int = 5,
):
    eval_is_win_buffer = []
    eval_reward_buffer = []
    eval_steps_buffer = []
    for _ in range(num_eval_episodes):
        agent.reset_agent()
        episode_reward = 0.0
        episode_step = 0
        terminated = False
        obs, state = env.reset()
        while not terminated:
            available_actions = env.get_available_actions()
            actions = agent.predict(obs, available_actions)
            obs, state, reward, terminated, info = env.step(actions)
            episode_step += 1
            episode_reward += reward

        is_win = env.win_counted

        eval_reward_buffer.append(episode_reward)
        eval_steps_buffer.append(episode_step)
        eval_is_win_buffer.append(is_win)

    eval_rewards = np.mean(eval_reward_buffer)
    eval_steps = np.mean(eval_steps_buffer)
    eval_win_rate = np.mean(eval_is_win_buffer)

    return eval_rewards, eval_steps, eval_win_rate
