import contextlib
import multiprocessing as mp
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from .base_vec_env import BaseVecEnv, CloudpickleWrapper


@contextlib.contextmanager
def clear_mpi_env_vars():
    """from mpi4py import MPI will call MPI_Init by default.

    If the child process has MPI environment variables, MPI will think that the
    child process is an MPI process just like the parent and do bad things such
    as hang. This context manager is a hacky way to clear those environment
    variables temporarily such as when we are starting multiprocessing
    Processes.
    """
    removed_environment = {}
    for k, v in list(os.environ.items()):
        for prefix in ['OMPI_', 'PMI_']:
            if k.startswith(prefix):
                removed_environment[k] = v
                del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(removed_environment)


def worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
) -> None:
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                state, obs, reward, terminated, truncated, info = env.step(
                    data)
                remote.send((state, obs, reward, terminated, truncated, info))
            elif cmd == 'reset':
                state, obs, available_actions, reset_info = env.reset()
                remote.send((state, obs, available_actions, reset_info))
            elif cmd == 'render':
                remote.send(env.render())
            elif cmd == 'close':
                env.close()
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((
                    env.obs_space,
                    env.share_obs_space,
                    env.action_space,
                ))
            elif cmd == 'get_num_agents':
                remote.send((env.num_agents))
            else:
                raise NotImplementedError(
                    f'`{cmd}` is not implemented in the worker')
        except EOFError:
            break


class SubprocVecEnv(BaseVecEnv):
    """Creates a multiprocess vectorized wrapper for multiple environments,
    distributing each environment to its own process, allowing significant
    speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: Environments to run in subprocesses
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    :param in_series: number of environments to run in series in a single process
            (e.g. when len(env_fns) == 12 and in_series == 3, it will run 4 processes, each running 3 envs in series)
    """

    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        start_method: Optional[str] = 'spawn',
    ) -> None:
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)
        print('Num envs:', self.num_envs)

        if start_method is None:
            forkserver_available = 'forkserver' in mp.get_all_start_methods()
            start_method = 'forkserver' if forkserver_available else 'spawn'
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(self.num_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes,
                                               env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            process = ctx.Process(target=worker, args=args, daemon=True)
            self.processes.append(process)
        for process in self.processes:
            with clear_mpi_env_vars():
                process.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.num_agents = self.remotes[0].recv()
        obs_space, share_obs_space, action_space = self.remotes[0].recv()

        super().__init__(len(env_fns), obs_space, share_obs_space,
                         action_space)

    def step_async(self, actions: np.ndarray) -> None:
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(
        self,
    ) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray], Tuple[
            np.ndarray, ...]], np.ndarray, np.ndarray, List[Dict], ]:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        state, obs, rews, dones, infos = zip(*results)
        return (
            np.stack(state),
            np.stack(obs),
            np.stack(rews),
            np.stack(dones),
            infos,
        )

    def reset(
        self
    ) -> Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]]:
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        state, obs = zip(*results)
        return (np.stack(state), np.stack(obs))

    def close(self) -> None:
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


def shareworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, s_ob, reward, done, info, available_actions = env.step(data)
            if 'bool' in done.__class__.__name__:
                if done:
                    ob, s_ob, available_actions = env.reset()
            else:
                if np.all(done):
                    ob, s_ob, available_actions = env.reset()
            remote.send((ob, s_ob, reward, done, info, available_actions))
        elif cmd == 'reset':
            ob, s_ob, available_actions = env.reset()
            remote.send((ob, s_ob, available_actions))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.obs_space, env.share_obs_space, env.action_space))
        elif cmd == 'get_num_agents':
            remote.send((env.num_agents))
        else:
            raise NotImplementedError


class ShareSubprocVecEnv(BaseVecEnv):
    """Creates a multiprocess vectorized wrapper for multiple environments,
    distributing each environment to its own process, allowing significant
    speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: Environments to run in subprocesses
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    :param in_series: number of environments to run in series in a single process
            (e.g. when len(env_fns) == 12 and in_series == 3, it will run 4 processes, each running 3 envs in series)
    """

    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        start_method: Optional[str] = 'spawn',
    ) -> None:
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)

        if start_method is None:
            forkserver_available = 'forkserver' in mp.get_all_start_methods()
            start_method = 'forkserver' if forkserver_available else 'spawn'
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(self.num_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes,
                                               env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            process = ctx.Process(target=shareworker, args=args, daemon=True)
            self.processes.append(process)
        for process in self.processes:
            with clear_mpi_env_vars():
                process.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        obs_space, share_obs_space, action_space = self.remotes[0].recv()
        self.num_agents = self.remotes[0].recv()

        super().__init__(len(env_fns), obs_space, share_obs_space,
                         action_space)

    def step_async(self, actions: np.ndarray) -> None:
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(
        self,
    ) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray], Tuple[
            np.ndarray, ...]], np.ndarray, np.ndarray, List[Dict], ]:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        state, obs, rews, dones, infos = zip(*results)
        return (
            np.stack(state),
            np.stack(obs),
            np.stack(rews),
            np.stack(dones),
            infos,
        )

    def reset(
        self
    ) -> Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]]:
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        state, obs = zip(*results)
        return (np.stack(state), np.stack(obs))

    def close(self) -> None:
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


def chooseworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, s_ob, reward, done, info, available_actions = env.step(data)
            remote.send((ob, s_ob, reward, done, info, available_actions))
        elif cmd == 'reset':
            ob, s_ob, available_actions = env.reset(data)
            remote.send((ob, s_ob, available_actions))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.obs_space, env.share_obs_space, env.action_space))
        elif cmd == 'get_num_agents':
            remote.send((env.num_agents))
        else:
            raise NotImplementedError


class ChooseSubprocVecEnv(BaseVecEnv):
    """Creates a multiprocess vectorized wrapper for multiple environments,
    distributing each environment to its own process, allowing significant
    speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: Environments to run in subprocesses
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    :param in_series: number of environments to run in series in a single process
            (e.g. when len(env_fns) == 12 and in_series == 3, it will run 4 processes, each running 3 envs in series)
    """

    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        start_method: Optional[str] = 'spawn',
    ) -> None:
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)

        if start_method is None:
            forkserver_available = 'forkserver' in mp.get_all_start_methods()
            start_method = 'forkserver' if forkserver_available else 'spawn'
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(self.num_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes,
                                               env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            process = ctx.Process(target=chooseworker, args=args, daemon=True)
            self.processes.append(process)
        for process in self.processes:
            with clear_mpi_env_vars():
                process.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        obs_space, share_obs_space, action_space = self.remotes[0].recv()
        self.num_agents = self.remotes[0].recv()

        super().__init__(len(env_fns), obs_space, share_obs_space,
                         action_space)

    def step_async(self, actions: np.ndarray) -> None:
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(
        self,
    ) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray], Tuple[
            np.ndarray, ...]], np.ndarray, np.ndarray, List[Dict], ]:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        state, obs, rews, dones, infos = zip(*results)
        return (
            np.stack(state),
            np.stack(obs),
            np.stack(rews),
            np.stack(dones),
            infos,
        )

    def reset(
        self
    ) -> Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]]:
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        state, obs = zip(*results)
        return (np.stack(state), np.stack(obs))

    def close(self) -> None:
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
