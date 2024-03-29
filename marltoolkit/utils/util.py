"""Modified from https://github.com/DLR-RM/stable-
baselines3/stable_baselines3/common/vec_env/base_vec_env.py code to work with
multi-agent envs."""

from typing import List, Sequence, Tuple, Union

import numpy as np
from gymnasium import spaces


def tile_images(
        images_nhwc: Sequence[np.ndarray]) -> np.ndarray:  # pragma: no cover
    """Tile N images into one big PxQ image (P,Q) are chosen to be as close as
    possible, and if N is square, then P=Q.

    :param images_nhwc: list or array of images, ndim=4 once turned into array.
        n = batch index, h = height, w = width, c = channel
    :return: img_HWc, ndim=3
    """
    img_nhwc = np.asarray(images_nhwc)
    n_images, height, width, n_channels = img_nhwc.shape
    # new_height was named H before
    new_height = int(np.ceil(np.sqrt(n_images)))
    # new_width was named W before
    new_width = int(np.ceil(float(n_images) / new_height))
    img_nhwc = np.array(
        list(img_nhwc) +
        [img_nhwc[0] * 0 for _ in range(n_images, new_height * new_width)])
    # img_HWhwc
    out_image = img_nhwc.reshape(
        (new_height, new_width, height, width, n_channels))
    # img_HhWwc
    out_image = out_image.transpose(0, 2, 1, 3, 4)
    # img_Hh_Ww_c
    out_image = out_image.reshape(
        (new_height * height, new_width * width, n_channels))
    return out_image


def combined_shape(
        length: int,
        shape: Union[None, int, Tuple[int, ...]] = None) -> Tuple[int, ...]:
    """Expand the original shape.

    Parameters:
    - length (int): The length of the first dimension to expand.
    - shape (Union[None, int, Tuple[int, ...]]): The target shape to be expanded.

    Returns:
    - Tuple[int, ...]: A new shape that is expanded from the original shape.

    Examples:
    --------
    >>> length = 2
    >>> shape_1 = None
    >>> shape_2 = 3
    >>> shape_3 = (4, 5)
    >>> combined_shape(length, shape_1)
    (2,)
    >>> combined_shape(length, shape_2)
    (2, 3)
    >>> combined_shape(length, shape_3)
    (2, 4, 5)
    """
    if shape is None:
        return (length, )
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def flatten_list(lst: Union[List, Tuple]) -> List:
    """Flatten a list of lists or tuples.

    Parameters:
    - lst (Union[List, Tuple]): The input list or tuple containing nested lists or tuples.

    Returns:
    - List: A flattened list containing all elements from the input.

    Raises:
    - AssertionError: If the input is not a list or tuple, has length <= 0, or contains sublists/tuples with length <= 0.

    Examples:
    --------
    >>> lst1 = [[1, 2, 3], [4, 5], [6]]
    >>> flatten_list(lst1)
    [1, 2, 3, 4, 5, 6]

    >>> lst2 = [(1, 2), (3, 4), (5, 6)]
    >>> flatten_list(lst2)
    [1, 2, 3, 4, 5, 6]
    """
    # Check input conditions
    assert isinstance(lst, (list, tuple)), 'Input must be a list or tuple'
    assert len(lst) > 0, 'Input must have length > 0'
    assert all(len(sub_lst) > 0
               for sub_lst in lst), 'Sublists/tuples must have length > 0'

    # Flatten the list
    return [item for sublist in lst for item in sublist]


def check_for_nested_spaces(obs_space: spaces.Space) -> None:
    """Make sure the observation space does not have nested spaces
    (Dicts/Tuples inside Dicts/Tuples). If so, raise an Exception informing
    that there is no support for this.

    :param obs_space: an observation space
    """
    if isinstance(obs_space, (spaces.Dict, spaces.Tuple)):
        sub_spaces = (obs_space.spaces.values() if isinstance(
            obs_space, spaces.Dict) else obs_space.spaces)
        for sub_space in sub_spaces:
            if isinstance(sub_space, (spaces.Dict, spaces.Tuple)):
                raise NotImplementedError(
                    'Nested observation spaces are not supported (Tuple/Dict space inside Tuple/Dict space).'
                )
