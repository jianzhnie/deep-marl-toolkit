import multiprocessing as mp
import warnings
from collections import OrderedDict
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Tuple, Type, Union)

import gymnasium as gym
import numpy as np

from .base_vec_env import BaseVecEnv, CloudpickleWrapper
from .utils import is_wrapped


def _flatten_obs(
    obs: Union[List[Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray,
                                                                   ...]]],
               Tuple[Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray,
                                                                    ...]]], ],
    space: gym.spaces.Space,
) -> Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]]:
    """Flatten observations, depending on the observation space.

    :param obs: observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return: flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(
        obs, (list,
              tuple)), 'expected list or tuple of observations per environment'
    assert len(obs) > 0, 'need observations from at least one environment'

    if isinstance(space, gym.spaces.Dict):
        assert isinstance(
            space.spaces,
            OrderedDict), 'Dict space must have ordered subspaces'
        assert isinstance(
            obs[0], dict
        ), 'non-dict observation for environment with Dict observation space'
        return OrderedDict([(k, np.stack([o[k] for o in obs]))
                            for k in space.spaces.keys()])
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(
            obs[0], tuple
        ), 'non-tuple observation for environment with Tuple observation space'
        obs_len = len(space.spaces)
        return tuple(np.stack([o[i] for o in obs])
                     for i in range(obs_len))  # type: ignore[index]
    else:
        return np.stack(obs)  # type: ignore[arg-type]


def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
) -> None:
    # Import here to avoid a circular import
    parent_remote.close()
    env = env_fn_wrapper.x()
    reset_info: Optional[Dict[str, Any]] = {}
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                state, obs, reward, terminated, truncated, info = env.step(
                    data)
                done = terminated or truncated
                info['TimeLimit.truncated'] = truncated and not terminated
                if done:
                    # save final obs where user can get it, then reset
                    info['terminal_obs'] = obs
                    state, obs, reset_info = env.reset()
                remote.send((state, obs, reward, done, info, reset_info))
            elif cmd == 'reset':
                maybe_options = {'options': data[1]} if data[1] else {}
                state, obs, reset_info = env.reset(seed=data[0],
                                                   **maybe_options)
                remote.send((state, obs, reset_info))
            elif cmd == 'render':
                remote.send(env.render())
            elif cmd == 'close':
                env.close()
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send(
                    (env.obs_space, env.share_obs_space, env.action_space))
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
    """

    def __init__(self,
                 env_fns: List[Callable[[], gym.Env]],
                 start_method: Optional[str] = None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = 'forkserver' in mp.get_all_start_methods()
            start_method = 'forkserver' if forkserver_available else 'spawn'
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes,
                                               env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args,
                                  daemon=True)  # type: ignore[attr-defined]
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(('get_spaces', None))
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
        state, obs, rews, dones, infos, self.reset_infos = zip(
            *results)  # type: ignore[assignment]
        return (
            _flatten_obs(state, self.share_obs_space),
            _flatten_obs(obs, self.obs_space),
            np.stack(rews),
            np.stack(dones),
            infos,
        )  # type: ignore[return-value]

    def reset(
        self
    ) -> Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]]:
        for env_idx, remote in enumerate(self.remotes):
            remote.send(
                ('reset', (self._seeds[env_idx], self._options[env_idx])))
        results = [remote.recv() for remote in self.remotes]
        state, obs, self.reset_infos = zip(
            *results)  # type: ignore[assignment]
        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return _flatten_obs(state, self.share_obs_space), _flatten_obs(
            obs, self.obs_space)

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
