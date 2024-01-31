"""Modified from OpenAI Baselines code to work with multi-agent envs."""
from abc import ABC, abstractmethod
from multiprocessing import Pipe, Process
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from marltoolkit.utils.util import tile_images

from .base_vec_env import CloudpickleWrapper


class ShareVecEnv(ABC):
    """An abstract asynchronous, vectorized environment.

    Used to batch data from multiple copies of an environment, so that each
    observation becomes a batch of observations, and expected action is a batch
    of actions to be applied per-environment.
    """

    closed: bool = False
    viewer: Any = None

    metadata: Dict[str, List[str]] = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, num_envs: int, observation_space: Any,
                 share_observation_space: Any, action_space: Any):
        """Initialize the ShareVecEnv.

        Args:
            num_envs (int): Number of environments in the vectorized environment.
            observation_space: The observation space of a single environment.
            share_observation_space: The shared observation space.
            action_space: The action space of a single environment.
        """
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Reset all the environments and return an array of observations, or a
        dict of observation arrays.

        If step_async is still doing work, that work will
        be canceled, and step_wait() should not be called
        until step_async() is invoked again.

        Returns:
            Union[np.ndarray, Dict[str, np.ndarray]]: Observations after reset.
        """
        pass

    @abstractmethod
    def step_async(self, actions: Union[np.ndarray, List[Any]]) -> None:
        """Tell all the environments to start taking a step with the given
        actions. Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.

        Args:
            actions: Actions to take in each environment.
        """
        pass

    @abstractmethod
    def step_wait(
        self
    ) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], np.ndarray,
               np.ndarray, List[Dict[str, Any]]]:
        """Wait for the step taken with step_async().

        Returns:
            Tuple[Union[np.ndarray, Dict[str, np.ndarray]], np.ndarray, np.ndarray, List[Dict[str, Any]]]:
                - obs: Observations after the step.
                - rews: Rewards obtained from the step.
                - dones: "Episode done" booleans for each environment.
                - infos: A list of info objects for each environment.
        """
        pass

    def close_extras(self) -> None:
        """Clean up the extra resources, beyond what's in this base class.

        Only runs when not self.closed.
        """
        pass

    def close(self) -> None:
        """Close the ShareVecEnv.

        If the environment is already closed, do nothing.
        """
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(
        self, actions: Union[np.ndarray, List[Any]]
    ) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], np.ndarray,
               np.ndarray, List[Dict[str, Any]]]:
        """Step the environments synchronously.

        This is available for backward compatibility.

        Args:
            actions: Actions to take in each environment.

        Returns:
            Tuple[Union[np.ndarray, Dict[str, np.ndarray]], np.ndarray, np.ndarray, List[Dict[str, Any]]]:
                - obs: Observations after the step.
                - rews: Rewards obtained from the step.
                - dones: "Episode done" booleans for each environment.
                - infos: A list of info objects for each environment.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode: str = 'human') -> Any:
        """Render the vectorized environment.

        Args:
            mode (str): Rendering mode.

        Returns:
            Any: Rendering output based on the mode.
        """

        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == 'human':
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self) -> List[np.ndarray]:
        """Return RGB images from each environment.

        Returns:
            List[np.ndarray]: List of RGB images.
        """
        raise NotImplementedError


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if 'bool' in done.__class__.__name__:
                if done:
                    ob = env.reset()
            else:
                if np.all(done):
                    ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send((ob))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.share_observation_space,
                         env.action_space))
        elif cmd == 'get_num_agents':
            remote.send((env.num_agents))
        else:
            raise NotImplementedError


class SubprocVecEnv(ShareVecEnv):

    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(
                target=worker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote,
                 env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for p in self.ps:
            p.daemon = (
                True  # if the main process crashes, we should not cause things to hang
            )
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[
            0].recv()
        self.remotes[0].send(('get_num_agents', None))
        self.num_agents = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
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
            remote.send((env.observation_space, env.share_observation_space,
                         env.action_space))
        elif cmd == 'get_num_agents':
            remote.send((env.num_agents))
        else:
            raise NotImplementedError


class ShareSubprocVecEnv(ShareVecEnv):

    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(
                target=shareworker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn)),
            ) for (work_remote, remote,
                   env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for p in self.ps:
            p.daemon = (
                True  # if the main process crashes, we should not cause things to hang
            )
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[
            0].recv()
        self.remotes[0].send(('get_num_agents', None))
        self.num_agents = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, share_obs, rews, dones, infos, available_actions = zip(*results)
        return (
            np.stack(obs),
            np.stack(share_obs),
            np.stack(rews),
            np.stack(dones),
            infos,
            np.stack(available_actions),
        )

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(available_actions)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
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
            remote.send((env.observation_space, env.share_observation_space,
                         env.action_space))
        elif cmd == 'get_num_agents':
            remote.send((env.num_agents))
        else:
            raise NotImplementedError


class ChooseSubprocVecEnv(ShareVecEnv):

    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(
                target=chooseworker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn)),
            ) for (work_remote, remote,
                   env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for p in self.ps:
            p.daemon = (
                True  # if the main process crashes, we should not cause things to hang
            )
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[
            0].recv()
        self.remotes[0].send(('get_num_agents', None))
        self.num_agents = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, share_obs, rews, dones, infos, available_actions = zip(*results)
        return (
            np.stack(obs),
            np.stack(share_obs),
            np.stack(rews),
            np.stack(dones),
            infos,
            np.stack(available_actions),
        )

    def reset(self, reset_choose):
        for remote, choose in zip(self.remotes, reset_choose):
            remote.send(('reset', choose))
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(available_actions)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


# single env


class DummyVecEnv(ShareVecEnv):

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(
            self,
            len(env_fns),
            env.observation_space,
            env.share_observation_space,
            env.action_space,
        )
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))

        self.actions = None
        return obs, rews, dones, infos

    def reset(self):
        obs = [env.reset() for env in self.envs]
        return np.array(obs)

    def close(self):
        for env in self.envs:
            env.close()


class ShareDummyVecEnv(ShareVecEnv):

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(
            self,
            len(env_fns),
            env.observation_space,
            env.share_observation_space,
            env.action_space,
        )
        self.actions = None
        self.num_agents = env.num_agents

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, share_obs, rews, dones, infos, available_actions = map(
            np.array, zip(*results))
        return obs, share_obs, rews, dones, infos, available_actions

    def reset(self):
        results = [env.reset() for env in self.envs]
        obs, share_obs, available_actions = map(np.array, zip(*results))
        return obs, share_obs, available_actions

    def close(self):
        for env in self.envs:
            env.close()


class ChooseDummyVecEnv(ShareVecEnv):

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(
            self,
            len(env_fns),
            env.observation_space,
            env.share_observation_space,
            env.action_space,
        )
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, share_obs, rews, dones, infos, available_actions = map(
            np.array, zip(*results))
        self.actions = None
        return obs, share_obs, rews, dones, infos, available_actions

    def reset(self, reset_choose):
        results = [
            env.reset(choose) for (env, choose) in zip(self.envs, reset_choose)
        ]
        obs, share_obs, available_actions = map(np.array, zip(*results))
        return obs, share_obs, available_actions

    def close(self):
        for env in self.envs:
            env.close()
