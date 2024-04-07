import sys

sys.path.append('../')
import torch
from torch.distributions import Categorical

from marltoolkit.envs.smacv1.smac_env import SMACWrapperEnv

if __name__ == '__main__':
    env = SMACWrapperEnv(map_name='3m')
    env_info = env.get_env_info()
    print('env_info:', env_info)

    action_dim = env_info['n_actions']
    num_agents = env_info['num_agents']
    results = env.reset()
    print('Reset:', results)

    avail_actions = env.get_available_actions()
    print('avail_actions:', avail_actions)
    available_actions = torch.tensor(avail_actions)
    actions_dist = Categorical(available_actions)
    random_actions = actions_dist.sample().numpy().tolist()

    print('random_actions: ', random_actions)
    results = env.step(random_actions)
    print('Step:', results)
    env.close()
