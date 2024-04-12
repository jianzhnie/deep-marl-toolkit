import logging
import os
from collections import OrderedDict

from .logging import get_logger

try:
    import wandb
except ImportError:
    pass


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name='marltoolkit',
                        log_file=log_file,
                        log_level=log_level)

    return logger


def get_outdir(path, *paths, inc=False):
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif inc:
        count = 1
        outdir_inc = outdir + '-' + str(count)
        while os.path.exists(outdir_inc):
            count = count + 1
            outdir_inc = outdir + '-' + str(count)
            assert count < 100
        outdir = outdir_inc
        os.makedirs(outdir)
    return outdir


def avg_val_from_list_of_dicts(list_of_dicts):
    sum_values = OrderedDict()
    count_dicts = OrderedDict()

    # Transpose the list of dictionaries into a list of key-value pairs
    for dictionary in list_of_dicts:
        for key, value in dictionary.items():
            sum_values[key] += value
            count_dicts[key] += 1

    # Calculate the average values using a dictionary comprehension
    avg_val_dict = {
        key: sum_value / count_dicts[key]
        for key, sum_value in sum_values.items()
    }

    return avg_val_dict


def update_summary(train_metrics, eval_metrics, log_wandb=False):
    rowd = OrderedDict()
    rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])
    if log_wandb:
        wandb.log(rowd)
