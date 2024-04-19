import sys

import numpy as np
import torch

sys.path.append('../')

from configs.arguments import get_common_args
from marltoolkit.agents.mappo_agent import MAPPOAgent
from marltoolkit.data.shared_buffer import SharedReplayBuffer
from marltoolkit.envs.smacv1 import SMACWrapperEnv


def _t2n(x):
    return x.detach().cpu().numpy()


class SMACRunner:
    """Runner class to perform training, evaluation.

    and data collection for SMAC. See parent class for details.
    """

    def __init__(self, args) -> None:
        #
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
                'u are choosing to use ippo, we set use_centralized_v to be False'
            )
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
            args.device = torch.device('cuda:0')
            torch.set_num_threads(args.n_training_threads)
            if args.cuda_deterministic:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
        else:
            print('choose to use cpu...')
            args.device = torch.device('cpu')
            torch.set_num_threads(args.n_training_threads)

        # parameters
        self.algorithm_name = args.algorithm_name
        self.use_centralized_v = args.use_centralized_v
        self.total_steps = args.total_steps
        self.num_envs = args.num_train_envs
        self.rnn_layers = args.rnn_layers
        self.hidden_size = args.hidden_size

        # envs
        self.env = SMACWrapperEnv(map_name=args.map_name, args=args)
        self.num_agents = self.env.num_agents
        args.episode_limit = self.env.episode_limit
        args.action_space = self.env.action_space
        args.actor_input_dim = self.env.get_actor_input_dim()
        args.state_dim = self.env.state_dim
        args.obs_shape = self.env.get_actor_input_shape()

        print('obs_shape: ', self.env.obs_shape)
        print('state_shape: ', self.env.state_shape)
        print('action_shape: ', self.env.action_shape)

        # policy network
        self.agent = MAPPOAgent(args)
        # buffer
        self.buffer: SharedReplayBuffer = SharedReplayBuffer(
            args.num_train_envs,
            self.env.num_agents,
            self.env.episode_limit,
            args.obs_shape,
            self.env.state_shape,
            self.env.action_shape,
            self.env.reward_shape,
            self.env.done_shape,
            args,
        )
        print(self.buffer.obs.shape)

    def run(self, args):
        """Collect training data, perform training updates, and evaluate
        policy."""

        self.warmup()
        episodes = int(self.total_steps) // args.episode_limit // self.num_envs
        for episode in range(episodes):
            for step in range(args.episode_limit):
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states_actor,
                    rnn_states_critic,
                ) = self.collect(step)
                # Obser reward and next obs
                obs, state, rewards, terminated, truncated, infos = self.env.step(
                    actions)
                available_actions = self.env.get_available_actions()
                dones = terminated or truncated

                data = (
                    obs,
                    state,
                    rewards,
                    actions,
                    dones,
                    infos,
                    values,
                    action_log_probs,
                    available_actions,
                    rnn_states_actor,
                    rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()

    def warmup(self):
        """Collect warmup pre-training data."""

        # reset env
        obs, state, info = self.env.reset()
        available_actions = self.env.get_available_actions()

        # replay buffer
        if not self.use_centralized_v:
            state = obs

        self.buffer.state[0] = state.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        """Collect rollouts for training."""

        self.agent.prep_rollout()
        data = [
            self.buffer.obs[step],
            self.buffer.state[step],
            self.buffer.masks[step],
            self.buffer.available_actions[step],
            self.buffer.rnn_states_actor[step],
            self.buffer.rnn_states_critic[step],
        ]
        for idx in range(len(data)):
            print(data[idx].shape)
        for idx in range(len(data)):
            data[idx] = torch.tensor(data[idx]).to(args.device)

        value, action, action_log_prob, rnn_state, rnn_state_critic = (
            self.agent.get_actions(*data))
        # [self.env, agents, dim]
        values = np.array(np.split(_t2n(value), self.num_envs))
        actions = np.array(np.split(_t2n(action), self.num_envs))
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.num_envs))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.num_envs))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_state_critic), self.num_envs))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        """Insert data into buffer.

        :param data: (Tuple) data to insert into training buffer.
        """
        (
            obs,
            state,
            rewards,
            actions,
            dones,
            infos,
            values,
            action_log_probs,
            available_actions,
            rnn_states_actor,
            rnn_states_critic,
        ) = data

        dones_env = np.all(dones, axis=1)

        rnn_states_actor[dones_env is True] = np.zeros(
            (
                (dones_env is True).sum(),
                self.num_agents,
                self.rnn_layers,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        rnn_states_critic[dones_env is True] = np.zeros(
            (
                (dones_env is True).sum(),
                self.num_agents,
                *self.buffer.rnn_states_critic.shape[3:],
            ),
            dtype=np.float32,
        )

        masks = np.ones((self.num_envs, self.num_agents, 1), dtype=np.float32)
        masks[dones_env is True] = np.zeros(
            ((dones_env is True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.num_envs, self.num_agents, 1),
                               dtype=np.float32)
        active_masks[dones is True] = np.zeros(((dones is True).sum(), 1),
                                               dtype=np.float32)
        active_masks[dones_env is True] = np.ones(
            ((dones_env is True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array(
            [[[0.0] if info[agent_id]['bad_transition'] else [1.0]
              for agent_id in range(self.num_agents)] for info in infos])

        if not self.use_centralized_v:
            state = obs

        self.buffer.store_transitions(
            obs,
            state,
            actions,
            rewards,
            masks,
            bad_masks,
            active_masks,
            values,
            action_log_probs,
            available_actions,
            rnn_states_actor,
            rnn_states_critic,
        )

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.agent.prep_rollout()
        if self.algorithm_name == 'mat' or self.algorithm_name == 'mat_dec':
            next_values = self.agent.get_values(
                np.concatenate(self.buffer.state[-1]),
                np.concatenate(self.buffer.obs[-1]),
                np.concatenate(self.buffer.rnn_states_critic[-1]),
                np.concatenate(self.buffer.masks[-1]),
            )
        else:
            next_values = self.agent.get_values(
                np.concatenate(self.buffer.state[-1]),
                np.concatenate(self.buffer.rnn_states_critic[-1]),
                np.concatenate(self.buffer.masks[-1]),
            )
        next_values = np.array(np.split(_t2n(next_values), self.num_envs))
        self.buffer.compute_returns(next_values, self.agent.value_normalizer)

    def train(self):
        """Train policies with data in buffer."""
        self.agent.prep_training()
        train_infos = self.agent.learn(self.buffer)
        self.buffer.after_update()
        return train_infos


if __name__ == '__main__':
    args = get_common_args()
    runner = SMACRunner(args)
    runner.run(args)
