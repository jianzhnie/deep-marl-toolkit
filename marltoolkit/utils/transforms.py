import numpy as np


class OneHotTransform(object):

    def __init__(self, out_dim):
        self.out_dim = out_dim

    def __call__(self, agent_id):
        assert agent_id < self.out_dim
        one_hot_id = np.zeros(self.out_dim, dtype='float32')
        one_hot_id[agent_id] = 1.0
        return one_hot_id
