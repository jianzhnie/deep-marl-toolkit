from marltoolkit.envs.smacv1 import SMACWrapperEnv
from marltoolkit.envs.vec_env import SubprocVecEnv


def make_vec_env(env_id: str,
                 num_train_envs: int = 10,
                 num_test_envs: int = 10,
                 **kwargs):

    def make_env():
        if env_id == 'SMAC-v1':
            env = SMACWrapperEnv(env_id, **kwargs)
        else:
            raise ValueError(f'Unknown environment: {env_id}')
        return env

    env = make_env()
    train_envs = SubprocVecEnv([make_env() for _ in range(num_train_envs)])
    test_envs = SubprocVecEnv([make_env() for _ in range(num_test_envs)])
    return env, train_envs, test_envs
