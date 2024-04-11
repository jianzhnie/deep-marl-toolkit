import argparse
from typing import Tuple

import numpy as np

from marltoolkit.agents.base_agent import BaseAgent
from marltoolkit.data.ma_buffer import ReplayBuffer
from marltoolkit.envs import MultiAgentEnv


def run_train_episode(
    env: MultiAgentEnv,
    agent: BaseAgent,
    rpm: ReplayBuffer,
    args: argparse.Namespace = None,
):
    episode_reward = 0.0
    episode_step = 0
    done = False
    agent.init_hidden_states(batch_size=1)
    (obs, state, info) = env.reset()

    agents_id_onehot = env.get_agents_id_one_hot()
    if args.use_last_action:
        last_actions = np.zeros((args.num_agents, args.n_actions),
                                dtype=np.float32)
        obs = np.concatenate([obs, last_actions], axis=-1)
    if args.use_agent_id_onehot:
        obs = np.concatenate([obs, agents_id_onehot], axis=-1)

    while not done:
        available_actions = env.get_available_actions()
        actions = agent.sample(obs, available_actions)
        last_actions = env.get_actions_one_hot(actions)
        next_obs, next_state, reward, terminated, truncated, info = env.step(
            actions)
        if args.use_last_action:
            next_obs = np.concatenate([next_obs, last_actions], axis=-1)
        if args.use_agent_id_onehot:
            next_obs = np.concatenate([next_obs, agents_id_onehot], axis=-1)
        done = terminated or truncated
        transitions = {
            'obs': obs,
            'state': state,
            'actions': actions,
            'last_actions': last_actions,
            'available_actions': available_actions,
            'rewards': reward,
            'dones': done,
            'filled': False,
        }

        rpm.store_transitions(transitions)
        obs, state = next_obs, next_state

        episode_reward += reward
        episode_step += 1

    # fill the episode
    for _ in range(episode_step, args.episode_limit):
        rpm.episode_data.fill_mask()

    # ReplayBuffer store the episode data
    rpm.store_episodes()
    is_win = env.win_counted

    mean_loss = []
    mean_td_error = []
    if rpm.size() > args.memory_warmup_size:
        for _ in range(args.update_learner_freq):
            batch = rpm.sample(args.batch_size)
            loss, td_error = agent.learn(batch)
            mean_loss.append(loss)
            mean_td_error.append(td_error)

    mean_loss = np.mean(mean_loss) if mean_loss else None
    mean_td_error = np.mean(mean_td_error) if mean_td_error else None

    return episode_reward, episode_step, is_win, mean_loss, mean_td_error


def run_evaluate_episode(
    env: MultiAgentEnv,
    agent: BaseAgent,
    num_eval_episodes: int = 5,
    args: argparse.Namespace = None,
) -> Tuple[float, float, float]:
    eval_is_win_buffer = []
    eval_reward_buffer = []
    eval_steps_buffer = []
    for _ in range(num_eval_episodes):
        agent.init_hidden_states(batch_size=1)
        episode_reward = 0.0
        episode_step = 0
        done = False
        obs, state, info = env.reset()

        agents_id_onehot = env.get_agents_id_one_hot()
        if args.use_last_action:
            last_actions = np.zeros((args.num_agents, args.n_actions),
                                    dtype=np.float32)
            obs = np.concatenate([obs, last_actions], axis=-1)
        if args.use_agent_id_onehot:
            obs = np.concatenate([obs, agents_id_onehot], axis=-1)

        while not done:
            available_actions = env.get_available_actions()
            actions = agent.predict(obs, available_actions)
            last_actions = env.get_actions_one_hot(actions)
            next_obs, next_state, reward, terminated, truncated, info = env.step(
                actions)

            if args.use_last_action:
                next_obs = np.concatenate([next_obs, last_actions], axis=-1)
            if args.use_agent_id_onehot:
                next_obs = np.concatenate([next_obs, agents_id_onehot],
                                          axis=-1)

            done = terminated or truncated
            obs = next_obs
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
