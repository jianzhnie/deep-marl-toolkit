'''
Author: jianzhnie@126.com
Date: 2022-09-01 15:17:32
LastEditors: jianzhnie@126.com
LastEditTime: 2022-09-01 15:17:35
Description:

Copyright (c) 2022 by jianzhnie, All Rights Reserved.
'''

from collections import Counter
from typing import List

import six

__all__ = ['PiecewiseScheduler', 'LinearDecayScheduler']


class PiecewiseScheduler(object):
    """Set hyper parameters by a predefined step-based scheduler."""

    def __init__(self, scheduler_list):
        """Piecewise scheduler of hyper parameter.

        Args:
            scheduler_list: list of (step, value) pair. E.g. [(0, 0.001), (10000, 0.0005)]
        """
        assert len(scheduler_list) > 0

        for i in six.moves.range(len(scheduler_list) - 1):
            assert scheduler_list[i][0] < scheduler_list[i + 1][0], \
                    'step of scheduler_list should be incremental.'

        self.scheduler_list = scheduler_list

        self.cur_index = 0
        self.cur_step = 0
        self.cur_value = self.scheduler_list[0][1]

        self.scheduler_num = len(self.scheduler_list)

    def step(self, step_num=1):
        """Step step_num and fetch value according to following rule:

        Given scheduler_list: [(step_0, value_0), (step_1, value_1), ..., (step_N, value_N)],
        function will return value_K which satisfying self.cur_step >= step_K and self.cur_step < step_K+1

        Args:
            step_num (int): number of steps (default: 1)
        """
        assert isinstance(step_num, int) and step_num >= 1
        self.cur_step += step_num

        if self.cur_index < self.scheduler_num - 1:
            if self.cur_step >= self.scheduler_list[self.cur_index + 1][0]:
                self.cur_index += 1
                self.cur_value = self.scheduler_list[self.cur_index][1]

        return self.cur_value


class LinearDecayScheduler(object):
    """Set hyper parameters by a step-based scheduler with linear decay
    values."""

    def __init__(self, start_value, max_steps):
        """Linear decay scheduler of hyper parameter. Decay value linearly
        until 0.

        Args:
            start_value (float): start value
            max_steps (int): maximum steps
        """
        assert max_steps > 0
        self.cur_step = 0
        self.max_steps = max_steps
        self.start_value = start_value

    def step(self, step_num=1):
        """Step step_num and fetch value according to following rule:

        return_value = start_value * (1.0 - (cur_steps / max_steps))

        Args:
            step_num (int): number of steps (default: 1)

        Returns:
            value (float): current value
        """
        assert isinstance(step_num, int) and step_num >= 1
        self.cur_step = min(self.cur_step + step_num, self.max_steps)

        value = self.start_value * (1.0 -
                                    ((self.cur_step * 1.0) / self.max_steps))

        return value


class MultiStepScheduler(object):
    """step learning rate scheduler."""

    def __init__(self,
                 start_value: float,
                 max_steps: int,
                 milestones: List = None,
                 decay_factor: float = 0.1):
        assert max_steps > 0
        assert isinstance(decay_factor, float)
        assert decay_factor > 0 and decay_factor < 1
        self.milestones = Counter(milestones)
        self.cur_value = start_value
        self.cur_step = 0
        self.max_steps = max_steps
        self.decay_factor = decay_factor

    def step(self, step_num=1):
        assert isinstance(step_num, int) and step_num >= 1
        self.cur_step = min(self.cur_step + step_num, self.max_steps)

        if self.cur_step not in self.milestones:
            return self.cur_value
        else:
            self.cur_value *= self.decay_factor**self.milestones[self.cur_step]

        return self.cur_value


if __name__ == '__main__':
    scheduler = MultiStepScheduler(100, 100, [50, 80], 0.5)
    for i in range(101):
        value = scheduler.step()
        print(value)
