"""Helpers for dealing with vectorized environments."""

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from marltoolkit.utils.util import check_for_nested_spaces


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
