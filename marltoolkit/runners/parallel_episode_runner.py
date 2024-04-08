import argparse

import numpy as np

from marltoolkit.agents import BaseAgent
from marltoolkit.data import MaEpisodeData, OffPolicyBufferRNN
from marltoolkit.envs import BaseVecEnv


def run_train_episode(
    envs: BaseVecEnv,
    agent: BaseAgent,
    rpm: OffPolicyBufferRNN,
    args: argparse.Namespace = None,
):
    agent.reset_agent()
    # reset the environment
    obs, state, info = envs.reset()
    num_envs = envs.num_envs
    env_dones = envs.buf_dones
    episode_step = 0
    episode_score = np.zeros(num_envs, dtype=np.float32)
    filled = np.zeros([args.episode_limit, num_envs], dtype=np.int32)
    episode_data = MaEpisodeData(
        num_envs,
        args.num_agents,
        args.episode_limit,
        args.obs_shape,
        args.state_shape,
        args.action_shape,
        args.reward_shape,
        args.done_shape,
    )
    while not env_dones.all():
        available_actions = envs.get_available_actions()
        # Get actions from the agent
        actions = agent.sample(obs, available_actions)
        # Environment step
        next_obs, next_state, rewards, env_dones, info = envs.step(actions)
        # Fill the episode buffer
        filled[episode_step, :] = np.ones(num_envs)
        transitions = dict(
            obs=obs,
            state=state,
            actions=actions,
            rewards=rewards,
            env_dones=env_dones,
            available_actions=available_actions,
        )
        # Store the transitions
        episode_data.store_transitions(transitions)
        # Check if the episode is done
        for env_idx in range(num_envs):
            if env_dones[env_idx]:
                # Fill the rest of the episode with zeros
                filled[episode_step, env_idx] = 0
                # Get the episode score from the info
                final_info = info['final_info']
                episode_score[env_idx] = final_info[env_idx]['episode_score']
                # Get the current available actions
                available_actions = envs.get_available_actions()
                terminal_data = (next_obs, next_state, available_actions,
                                 filled)
                # Finish the episode
                rpm.finish_path(env_idx, episode_step, *terminal_data)

        # Update the episode step
        episode_step += 1
        obs, state = next_obs, next_state
        # Store the episode data
        rpm.store_episodes(episode_data.episode_buffer)

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
