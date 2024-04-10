import multiprocessing as mp
from typing import Any, Callable, Iterable, List, Optional, Type, Union

import gymnasium as gym
import numpy as np
from gymnasium.vector.utils import clear_mpi_env_vars

from marltoolkit.envs.smacv1.smac_env import SMACWrapperEnv
from marltoolkit.envs.vec_env import BaseVecEnv, CloudpickleWrapper
from marltoolkit.envs.vec_env.utils import combined_shape, flatten_list


def worker(
    remote: mp.connection.Connection = None,
    parent_remote: mp.connection.Connection = None,
    env_fn_wrappers: CloudpickleWrapper = None,
) -> None:

    def step_env(env: SMACWrapperEnv, actions: Union[np.ndarray, List[Any]]):
        state, obs, reward, terminated, truncated, info = env.step(actions)
        return state, obs, reward, terminated, truncated, info

    parent_remote.close()
    envs: List[SMACWrapperEnv] = [
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
                remote.send(CloudpickleWrapper((envs[0].env_info)))
            else:
                raise NotImplementedError(
                    f'`{cmd}` is not implemented in the worker')
        except KeyboardInterrupt:
            print('SubprocVecEnv worker: got KeyboardInterrupt')
        finally:
            for env in envs:
                env.close()


class SubprocVecSMAC(BaseVecEnv):
    """VecEnv that runs multiple environments in parallel in subproceses and
    communicates with them via pipes.

    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """

    def __init__(
        self,
        env_fns: List[Callable[[], SMACWrapperEnv]],
        start_method: Optional[str] = 'spawn',
    ):
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
        env_info = self.remotes[0].recv().fn
        self.obs_space = env_info['obs_space']
        self.state_space = env_info['state_space']
        self.action_dim = self.n_actions = env_info['n_actions']
        self.action_space = env_info['action_space']
        super().__init__(num_envs, self.obs_space, self.state_space,
                         self.action_space)

        self.obs_dim = env_info['obs_shape']
        self.state_dim = env_info['state_shape']
        self.num_agents = env_info['num_agents']
        self.obs_shape = (self.num_agents, self.obs_dim)
        self.act_shape = (self.num_agents, self.action_dim)
        self.rew_shape = (self.num_agents, 1)
        self.dim_reward = self.num_agents

        self.buf_obs = np.zeros(combined_shape(self.num_envs, self.obs_shape),
                                dtype=np.float32)
        self.buf_state = np.zeros(combined_shape(self.num_envs,
                                                 self.state_dim),
                                  dtype=np.float32)
        self.buf_terminal = np.zeros((self.num_envs, 1), dtype=bool)
        self.buf_truncation = np.zeros((self.num_envs, 1), dtype=bool)
        self.buf_done = np.zeros((self.num_envs, ), dtype=bool)
        self.buf_reward = np.zeros((self.num_envs, ) + self.rew_shape,
                                   dtype=np.float32)
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
        state, obs, infos = zip(*results)
        self.buf_state, self.buf_obs, self.buf_info = (np.array(state),
                                                       np.array(obs),
                                                       list(infos))
        return self.buf_state.copy(), self.buf_obs.copy(), self.buf_info.copy()

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
            for idx_env, env_done, remote in zip(range(self.num_envs),
                                                 self.buf_done, self.remotes):
                if not env_done:
                    result = remote.recv()
                    state, obs, reward, terminal, truncated, infos = result
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

    def get_attr(self,
                 attr_name: str,
                 indices: Union[None, int, Iterable[int]] = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(
        self,
        attr_name: str,
        value: Any,
        indices: Union[None, int, Iterable[int]] = None,
    ) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('set_attr', (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: Union[None, int, Iterable[int]] = None,
        **method_kwargs,
    ) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(
                ('env_method', (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def env_is_wrapped(
        self,
        wrapper_class: Type[gym.Wrapper],
        indices: Union[None, int, Iterable[int]] = None,
    ) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('is_wrapped', wrapper_class))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(
            self, indices: Union[None, int, Iterable[int]]) -> List[Any]:
        """Get the connection object needed to communicate with the wanted envs
        that are in subprocesses.

        :param indices: refers to indices of envs.
        :return: Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]
