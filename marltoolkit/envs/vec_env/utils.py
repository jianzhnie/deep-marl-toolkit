"""Helpers for dealing with vectorized environments."""

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def unwrap_wrapper(env: gym.Env,
                   wrapper_class: Type[gym.Wrapper]) -> Optional[gym.Wrapper]:
    """Retrieve a ``VecEnvWrapper`` object by recursively searching.

    :param env: Environment to unwrap
    :param wrapper_class: Wrapper to look for
    :return: Environment unwrapped till ``wrapper_class`` if it has been wrapped with it
    """
    env_tmp = env
    while isinstance(env_tmp, gym.Wrapper):
        if isinstance(env_tmp, wrapper_class):
            return env_tmp
        env_tmp = env_tmp.env
    return None


def is_wrapped(env: gym.Env, wrapper_class: Type[gym.Wrapper]) -> bool:
    """Check if a given environment has been wrapped with a given wrapper.

    :param env: Environment to check
    :param wrapper_class: Wrapper class to look for
    :return: True if environment has been wrapped with ``wrapper_class``.
    """
    return unwrap_wrapper(env, wrapper_class) is not None


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


def flatten_obs(obs, space: gym.spaces.Space):
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


def copy_obs_dict(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Deep-copy a dict of numpy arrays.

    :param obs: a dict of numpy arrays.
    :return: a dict of copied numpy arrays.
    """
    assert isinstance(
        obs, OrderedDict), f"unexpected type for observations '{type(obs)}'"
    return OrderedDict([(k, np.copy(v)) for k, v in obs.items()])


def dict_to_obs(
    obs_space: spaces.Space, obs_dict: Dict[Any, np.ndarray]
) -> Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]]:
    """Convert an internal representation raw_obs into the appropriate type
    specified by space.

    :param obs_space: an observation space.
    :param obs_dict: a dict of numpy arrays.
    :return: returns an observation of the same type as space.
        If space is Dict, function is identity; if space is Tuple, converts dict to Tuple;
        otherwise, space is unstructured and returns the value raw_obs[None].
    """
    if isinstance(obs_space, spaces.Dict):
        return obs_dict
    elif isinstance(obs_space, spaces.Tuple):
        assert len(obs_dict) == len(
            obs_space.spaces
        ), 'size of observation does not match size of observation space'
        return tuple(obs_dict[i] for i in range(len(obs_space.spaces)))
    else:
        assert set(obs_dict.keys()) == {
            None
        }, 'multiple observation keys for unstructured observation space'
        return obs_dict[None]


def obs_space_info(
    obs_space: spaces.Space,
) -> Tuple[List[str], Dict[Any, Tuple[int, ...]], Dict[Any, np.dtype]]:
    """Get dict-structured information about a gym.Space.

    Dict spaces are represented directly by their dict of subspaces.
    Tuple spaces are converted into a dict with keys indexing into the tuple.
    Unstructured spaces are represented by {None: obs_space}.

    :param obs_space: an observation space
    :return: A tuple (keys, shapes, dtypes):
        keys: a list of dict keys.
        shapes: a dict mapping keys to shapes.
        dtypes: a dict mapping keys to dtypes.
    """
    check_for_nested_spaces(obs_space)
    if isinstance(obs_space, spaces.Dict):
        assert isinstance(
            obs_space.spaces,
            OrderedDict), 'Dict space must have ordered subspaces'
        subspaces = obs_space.spaces
    elif isinstance(obs_space, spaces.Tuple):
        subspaces = {i: space
                     for i, space in enumerate(obs_space.spaces)
                     }  # type: ignore[assignment]
    else:
        assert not hasattr(
            obs_space,
            'spaces'), f"Unsupported structured space '{type(obs_space)}'"
        subspaces = {None: obs_space}  # type: ignore[assignment]
    keys = []
    shapes = {}
    dtypes = {}
    for key, box in subspaces.items():
        keys.append(key)
        shapes[key] = box.shape
        dtypes[key] = box.dtype
    return keys, shapes, dtypes


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
