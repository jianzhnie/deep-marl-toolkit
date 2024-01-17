import contextlib
import multiprocessing as mp
import os
import warnings
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Tuple, Type, Union)

import gymnasium
import numpy as np
from stable_baselines3.common.vec_env.subproc_vec_env import _flatten_obs

from .vec_env import CloudpickleWrapper, VecEnv


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
    # Import here to avoid a circular import
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    env = env_fn_wrapper.x()
    reset_info: Optional[Dict[str, Any]] = {}
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                observation, reward, terminated, truncated, info = env.step(
                    data)
                # convert to SB3 VecEnv api
                done = terminated or truncated
                info['TimeLimit.truncated'] = truncated and not terminated
                if done:
                    # save final observation where user can get it, then reset
                    info['terminal_observation'] = observation
                    observation, reset_info = env.reset()
                remote.send((observation, reward, done, info, reset_info))
            elif cmd == 'reset':
                maybe_options = {'options': data[1]} if data[1] else {}
                observation, reset_info = env.reset(
                    seed=data[0], **maybe_options)
                remote.send((observation, reset_info))
            elif cmd == 'render':
                remote.send(env.render())
            elif cmd == 'close':
                env.close()
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'get_num_agents':
                remote.send((env.num_agents))
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                remote.send(setattr(
                    env, data[0], data[1]))  # type: ignore[func-returns-value]
            elif cmd == 'is_wrapped':
                remote.send(is_wrapped(env, data))
            else:
                raise NotImplementedError(
                    f'`{cmd}` is not implemented in the worker')
        except EOFError:
            break


class SubprocVecEnv(VecEnv):
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

    def __init__(self,
                 env_fns: List[Callable[[], gymnasium.Env]],
                 start_method: Optional[str] = 'spawn',
                 in_series: int = 1):
        self.waiting = False
        self.closed = False
        self.in_series = in_series
        n_envs = len(env_fns)

        assert n_envs % in_series == 0, 'Number of envs must be divisible by number of envs to run in series'
        self.nremotes = n_envs // in_series
        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = 'forkserver' in mp.get_all_start_methods()
            start_method = 'forkserver' if forkserver_available else 'spawn'
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(self.nremotes)])
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

        self.remotes[0].send(('get_spaces', None))
        self.num_agents = self.remotes[0].recv()
        observation_space, action_space = self.remotes[0].recv()

        super().__init__(len(env_fns), observation_space, action_space)

    def step_async(self, actions: np.ndarray) -> None:
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(
        self
    ) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray], Tuple[
            np.ndarray, ...]], np.ndarray, np.ndarray, List[Dict]]:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos, self.reset_infos = zip(*results)
        return _flatten_obs(
            obs,
            self.observation_space), np.stack(rews), np.stack(dones), infos

    def reset(
        self
    ) -> Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]]:
        for env_idx, remote in enumerate(self.remotes):
            remote.send(
                ('reset', (self._seeds[env_idx], self._options[env_idx])))
        results = [remote.recv() for remote in self.remotes]
        obs, self.reset_infos = zip(*results)
        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return _flatten_obs(obs, self.observation_space)

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

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        if self.render_mode != 'rgb_array':
            warnings.warn(
                f'The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images.'
            )
            return [None for _ in self.remotes]
        for pipe in self.remotes:
            # gather render return from subprocesses
            pipe.send(('render', None))
        outputs = [pipe.recv() for pipe in self.remotes]
        return outputs

    def get_attr(self,
                 attr_name: str,
                 indices: Union[None, int, Iterable[int]] = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self,
                 attr_name: str,
                 value: Any,
                 indices: Union[None, int, Iterable[int]] = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('set_attr', (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self,
                   method_name: str,
                   *method_args,
                   indices: Union[None, int, Iterable[int]] = None,
                   **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(
                ('env_method', (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def env_is_wrapped(
            self,
            wrapper_class: Type[gymnasium.Wrapper],
            indices: Union[None, int, Iterable[int]] = None) -> List[bool]:
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
