import multiprocessing as mp
from typing import Any, Callable, List, Optional, Union

import numpy as np
from gymnasium.spaces import Box, Discrete

from marltoolkit.envs.subproc_vec_env import clear_mpi_env_vars
from marltoolkit.envs.vec_env import CloudpickleWrapper, VecEnv
from marltoolkit.utils.util import combined_shape, flatten_list

from ..multiagentenv import MultiAgentEnv


def worker(
    remote: mp.connection.Connection = None,
    parent_remote: mp.connection.Connection = None,
    env_fn_wrappers: CloudpickleWrapper = None,
) -> None:

    def step_env(env: MultiAgentEnv, actions: Union[np.ndarray, List[Any]]):
        obs, state, reward_n, terminated, truncated, info = env.step(actions)
        return obs, state, reward_n, terminated, truncated, info

    parent_remote.close()
    envs: List[MultiAgentEnv] = [
        env_fn_wrapper() for env_fn_wrapper in env_fn_wrappers.x
    ]
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send(
                    [step_env(env, action) for env, action in zip(envs, data)])
            elif cmd == 'get_available_actions':
                remote.send([env.get_available_actions() for env in envs])
            elif cmd == 'reset':
                remote.send([env.reset() for env in envs])
            elif cmd == 'render':
                remote.send([env.render(data) for env in envs])
            elif cmd == 'close':
                remote.send([env.close() for env in envs])
                remote.close()
                break
            elif cmd == 'get_env_info':
                remote.send(
                    CloudpickleWrapper((envs[0].env_info, envs[0].n_enemies)))
            else:
                raise NotImplementedError(
                    f'`{cmd}` is not implemented in the worker')
        except KeyboardInterrupt:
            print('SubprocVecEnv worker: got KeyboardInterrupt')
        finally:
            for env in envs:
                env.close()


class SubprocVecEnvSC2(VecEnv):
    """VecEnv that runs multiple environments in parallel in subproceses and
    communicates with them via pipes.

    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """

    def __init__(self,
                 env_fns: List[Callable[[], MultiAgentEnv]],
                 start_method: Optional[str] = 'spawn'):
        """
        Arguments:
        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        (e.g. when len(env_fns) == 12 and in_series == 3, it will run 4 processes, each running 3 envs in series)
        """
        self.waiting = False
        self.closed = False
        self.n_remotes = num_envs = len(env_fns)
        env_fns = np.array_split(env_fns, self.n_remotes)
        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = 'forkserver' in mp.get_all_start_methods()
            start_method = 'forkserver' if forkserver_available else 'spawn'

        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(self.n_remotes)])

        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes,
                                               env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=worker, args=args, daemon=True)
            with clear_mpi_env_vars():
                process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(('get_env_info', None))
        env_info, self.num_enemies = self.remotes[0].recv().x
        self.obs_dim = env_info['obs_shape']
        self.action_dim = self.n_actions = env_info['n_actions']
        observation_space = (self.obs_dim, )
        action_space = (self.action_dim, )
        super().__init__(num_envs, observation_space, action_space)

        self.state_dim = env_info['state_shape']
        self.num_agents = env_info['n_agents']
        self.obs_shape = (self.num_agents, self.obs_dim)
        self.act_shape = (self.num_agents, self.action_dim)
        self.rew_shape = (self.num_agents, 1)
        self.dim_reward = self.num_agents
        self.action_space = Discrete(n=self.action_dim)
        self.state_space = Box(
            low=-np.inf, high=np.inf, shape=[self.state_dim])

        self.buf_obs = np.zeros(
            combined_shape(self.num_envs, self.obs_shape), dtype=np.float32)
        self.buf_state = np.zeros(
            combined_shape(self.num_envs, self.state_dim), dtype=np.float32)
        self.buf_terminal = np.zeros((self.num_envs, 1), dtype=bool)
        self.buf_truncation = np.zeros((self.num_envs, 1), dtype=bool)
        self.buf_done = np.zeros((self.num_envs, ), dtype=bool)
        self.buf_reward = np.zeros(
            (self.num_envs, ) + self.rew_shape, dtype=np.float32)
        self.buf_info = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.battles_game = np.zeros(self.num_envs, np.int32)
        self.battles_won = np.zeros(self.num_envs, np.int32)
        self.dead_allies_count = np.zeros(self.num_envs, np.int32)
        self.dead_enemies_count = np.zeros(self.num_envs, np.int32)
        self.max_episode_length = env_info['episode_limit']

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        results = flatten_list(results)
        state, obs, obs_concate, infos = zip(*results)
        self.buf_state, self.buf_obs, self.buf_info = (np.array(state),
                                                       np.array(obs),
                                                       list(infos))
        return self.buf_obs.copy(), self.buf_state.copy(), self.buf_info.copy()

    def step_async(self, actions: Union[np.ndarray, List[Any]]):
        self._assert_not_closed()
        actions = np.array_split(actions, self.n_remotes)
        for remote, action, env_done in zip(self.remotes, actions,
                                            self.buf_done):
            if not env_done:
                remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        if self.waiting:
            for idx_env, env_done, remote in zip(
                    range(self.num_envs), self.buf_done, self.remotes):
                if not env_done:
                    result = remote.recv()
                    state, obs, obs_concate, reward, terminal, truncated, infos = result
                    self.buf_obs[idx_env], self.buf_state[idx_env] = (
                        np.array(obs),
                        np.array(state),
                    )
                    self.buf_reward[idx_env], self.buf_terminal[idx_env] = (
                        np.array(reward),
                        np.array(terminal),
                    )
                    self.buf_truncation[idx_env], self.buf_info[idx_env] = (
                        np.array(truncated),
                        infos,
                    )

                    if (self.buf_terminal[idx_env].all()
                            or self.buf_truncation[idx_env].all()):
                        self.buf_done[idx_env] = True
                        self.battles_game[idx_env] += 1
                        if infos['battle_won']:
                            self.battles_won[idx_env] += 1
                        self.dead_allies_count[idx_env] += infos['dead_allies']
                        self.dead_enemies_count[idx_env] += infos[
                            'dead_enemies']
                else:
                    self.buf_terminal[idx_env,
                                      0], self.buf_truncation[idx_env, 0] = (
                                          False,
                                          False,
                                      )

        self.waiting = False
        return (
            self.buf_obs.copy(),
            self.buf_state.copy(),
            self.buf_reward.copy(),
            self.buf_terminal.copy(),
            self.buf_truncation.copy(),
            self.buf_info.copy(),
        )

    def close_extras(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

    def render(self, mode):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', mode))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def get_available_actions(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_available_actions', None))
        avail_actions = [remote.recv() for remote in self.remotes]
        return np.array(avail_actions)

    def _assert_not_closed(self):
        assert (not self.closed
                ), 'Trying to operate on a SubprocVecEnv after calling close()'

    def __del__(self):
        if not self.closed:
            self.close()
