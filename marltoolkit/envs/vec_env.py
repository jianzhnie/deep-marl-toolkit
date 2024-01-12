from abc import ABC, abstractmethod


class VecEnv(ABC):

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
        self.closed = False

    @abstractmethod
    def reset(self):
        """Reset all the environments and return an array of observations, or a
        dict of observation arrays.

        If step_async is still doing work, that work will be cancelled and
        step_wait() should not be called until step_async() is invoked again.
        """
        raise NotImplementedError

    @abstractmethod
    def step_async(self, actions):
        """Tell all the environments to start taking a step with the given
        actions.

        Call step_wait() to get the results of the step. You should not call
        this if a step_async run is already pending.
        """
        raise NotImplementedError

    @abstractmethod
    def step_wait(self):
        """Wait for the step taken with step_async(). Returns (obs, rews,
        dones, infos):

        - obs: an array of observations, or a dict of
               arrays of observations.
        - rews: an array of rewards
        - dones: an array of "episode done" booleans
        - infos: a sequence of info objects
        """
        raise NotImplementedError

    @abstractmethod
    def close_extras(self):
        """Clean up the  extra resources, beyond what's in this base class.

        Only runs when not self.closed.
        """
        raise NotImplementedError

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode):
        raise NotImplementedError

    def close(self):
        if self.closed:
            return
        self.close_extras()
        self.closed = True
