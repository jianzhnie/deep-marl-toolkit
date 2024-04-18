import argparse
from typing import Tuple

from marltoolkit.agents.base_agent import BaseAgent
from marltoolkit.data.ma_buffer import ReplayBuffer
from marltoolkit.envs import MultiAgentEnv
from marltoolkit.utils.logger.logs import avg_val_from_list_of_dicts


def run_train_episode(
    env: MultiAgentEnv,
    agent: BaseAgent,
    rpm: ReplayBuffer,
    args: argparse.Namespace = None,
) -> dict[str, float]:
    episode_reward = 0.0
    episode_step = 0
    done = False
    agent.init_hidden_states(batch_size=1)
    (obs, state, info) = env.reset()
    while not done:
        available_actions = env.get_available_actions()
        actions = agent.sample(obs=obs, available_actions=available_actions)
        next_obs, next_state, reward, terminated, truncated, info = env.step(
            actions)
        done = terminated or truncated
        transitions = {
            'obs': obs,
            'state': state,
            'actions': actions,
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

    train_res_lst = []
    if rpm.size() > args.memory_warmup_size:
        for _ in range(args.learner_update_freq):
            batch = rpm.sample(args.batch_size)
            results = agent.learn(batch)
            train_res_lst.append(results)

    train_res_dict = avg_val_from_list_of_dicts(train_res_lst)

    train_res_dict['episode_reward'] = episode_reward
    train_res_dict['episode_step'] = episode_step
    train_res_dict['win_rate'] = is_win
    return train_res_dict


def run_eval_episode(
    env: MultiAgentEnv,
    agent: BaseAgent,
    num_eval_episodes: int = 5,
    args: argparse.Namespace = None,
) -> Tuple[float, float, float]:
    eval_res_list = []
    for _ in range(num_eval_episodes):
        agent.init_hidden_states(batch_size=1)
        episode_reward = 0.0
        episode_step = 0
        done = False
        obs, state, info = env.reset()

        while not done:
            available_actions = env.get_available_actions()
            actions = agent.predict(
                obs=obs,
                available_actions=available_actions,
            )
            next_obs, next_state, reward, terminated, truncated, info = env.step(
                actions)
            done = terminated or truncated
            obs = next_obs
            episode_step += 1
            episode_reward += reward

        is_win = env.win_counted
        eval_res_list.append({
            'episode_reward': episode_reward,
            'episode_step': episode_step,
            'win_rate': is_win,
        })
    eval_res_dict = avg_val_from_list_of_dicts(eval_res_list)
    return eval_res_dict
