import argparse

import numpy as np

from marltoolkit.agents import BaseAgent
from marltoolkit.data import OffPolicyBufferRNN
from marltoolkit.envs import SubprocVecEnvSC2


def run_train_episode(
    envs: SubprocVecEnvSC2,
    agent: BaseAgent,
    rpm: OffPolicyBufferRNN,
    args: argparse.Namespace = None,
):

    episode_limit = args.episode_limit
    agent.reset_agent()
    episode_reward = 0.0
    episode_step = 0
    state, obs, info = envs.reset()
    envs_done = envs.buf_done
    num_envs = envs.num_envs
    filled = np.zeros([num_envs, episode_limit, 1], dtype=np.int32)
    while not envs_done.all():
        available_actions = envs.get_available_actions()
        actions = agent.sample(obs, available_actions)
        next_state, next_obs, reward, terminated, truncated, info = envs.step(
            actions)
        filled[:, episode_step] = np.ones([num_envs, 1])
        envs_done = envs.buf_done
        episode_reward += reward
        episode_step += 1
        obs = next_obs
        rpm.store_transitions()

        for env_idx in range(num_envs):
            if envs_done[env_idx]:
                filled[env_idx, episode_step, 0] = 0

            if terminated[env_idx] or truncated[env_idx]:
                pass
    rpm.store_episodes()
    mean_loss = []
    mean_td_error = []
    if rpm.size() > args.memory_warmup_size:
        for _ in range(args.update_learner_freq):
            batch = rpm.sample_batch(args.batch_size)
            loss, td_error = agent.learn(**batch)
            mean_loss.append(loss)
            mean_td_error.append(td_error)

    mean_loss = np.mean(mean_loss) if mean_loss else None
    mean_td_error = np.mean(mean_td_error) if mean_td_error else None

    return episode_reward, episode_step, mean_loss, mean_td_error


def run_evaluate_episode(
    env: SubprocVecEnvSC2,
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
        state, obs = env.reset()
        while not terminated:
            available_actions = env.get_available_actions()
            actions = agent.predict(obs, available_actions)
            state, obs, reward, terminated = env.step(actions)
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
