import argparse

from marltoolkit.envs.smacv1 import SMACWrapperEnv
from marltoolkit.envs.vec_env import SubprocVecEnv


def make_vec_env(
    env_id: str,
    map_name: str,
    num_train_envs: int = 10,
    num_test_envs: int = 10,
    **kwargs,
):

    def make_env():
        if env_id == 'SMAC-v1':
            env = SMACWrapperEnv(map_name, **kwargs)
        else:
            raise ValueError(f'Unknown environment: {env_id}')
        return env

    train_envs = SubprocVecEnv([make_env for _ in range(num_train_envs)])
    test_envs = SubprocVecEnv([make_env for _ in range(num_test_envs)])
    return train_envs, test_envs


def get_actor_input_dim(args: argparse.Namespace) -> None:
    """Get the input shape of the actor model.

    Args:
        args (argparse.Namespace): The arguments
    Returns:
        input_shape (int): The input shape of the actor model.
    """
    input_dim = args.obs_dim
    if args.use_global_state:
        input_dim += args.state_dim
    if args.use_last_actions:
        input_dim += args.n_actions
    if args.use_agents_id_onehot:
        input_dim += args.num_agents
    return input_dim


def get_critic_input_dim(args: argparse.Namespace) -> None:
    """Get the input shape of the critic model.

    Args:
        args (argparse.Namespace): The arguments.

    Returns:
        input_dim (int): The input shape of the critic model.
    """
    input_dim = args.obs_dim
    if args.use_global_state:
        input_dim += args.state_dim
    if args.use_last_actions:
        input_dim += args.n_actions
    if args.use_agents_id_onehot:
        input_dim += args.num_agents
    return input_dim


def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'Tuple':
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape


def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == 'MultiDiscrete':
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == 'Box':
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == 'MultiBinary':
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1
    return act_shape
