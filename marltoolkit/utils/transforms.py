import numpy as np


class OneHotTransform(object):
    """One hot transform, convert index to one hot vector.

    Args:
        out_dim (int): The dimension of one hot vector.
    """

    def __init__(self, out_dim: int) -> None:
        self.out_dim = out_dim

    def __call__(self, index: int) -> np.ndarray:
        assert index < self.out_dim
        one_hot_id = np.zeros([self.out_dim])
        one_hot_id[index] = 1.0
        return one_hot_id
