import argparse

import numpy as np

from marltoolkit.agents import BaseAgent
from marltoolkit.data import OffPolicyBufferRNN
from marltoolkit.envs import BaseVecEnv


def run_train_episode(
    envs: BaseVecEnv,
    agent: BaseAgent,
    rpm: OffPolicyBufferRNN,
    args: argparse.Namespace = None,
):
    episode_limit = args.episode_limit
    agent.reset_agent()
    episode_step, best_score = 0, -np.inf
    obs, state, info = envs.reset()
    num_envs = envs.num_envs
    episode_score = np.zeros(num_envs, dtype=np.float32)
    filled = np.zeros([num_envs, episode_limit, 1], dtype=np.int32)
    env_dones = np.zeros((num_envs, ), dtype=np.bool_)
    while not env_dones.all():
        available_actions = envs.get_available_actions()
        actions = agent.sample(obs, available_actions)
        next_obs, next_states, rewards, env_dones, infos = envs.step(actions)
        filled[:, episode_step] = np.ones([num_envs, 1])
        episode_step += 1
        episode_score += rewards
        obs = next_obs
        transitions = (obs, state, actions, rewards, next_obs, next_states,
                       env_dones)
        rpm.store_transitions(transitions)
        for env_idx in range(num_envs):
            if env_dones[env_idx]:
                filled[env_idx, episode_step, 0] = 0
                if best_score < episode_score[-1]:
                    best_score = episode_score[-1]
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

    return episode_score, episode_step, mean_loss, mean_td_error


def run_evaluate_episode(
    env: BaseVecEnv,
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
