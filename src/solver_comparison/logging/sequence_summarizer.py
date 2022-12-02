import copy
from copy import deepcopy

import numpy as np


class OnlineSequenceSummary:
    """Returns N equally spaced samples from a sequence of length N.

    Use-case: We are running a process for an unknown number of steps,
    and would like to save the state a few times

    If we knew the process would take T=100 steps and want N=6 save points,
    we would save the value at T=0, 20, 40, 60, 80, 100.
    This utility approximates this when T is unknown.

    Args:
        n_to_save (int): Number of samples to save.
            The list will also separately save the first and last value seen.
    """

    def __init__(self, n_to_save: int):
        self.first = None
        self.last = None
        self.saved_values = [None] * n_to_save
        self.saved_iter = [None] * n_to_save
        self.iter_to_save = [i + 1 for i in range(n_to_save)]
        self.curr_iter = 0

    def update(self, value):
        if self.curr_iter == 0:
            self.first = deepcopy(value)
        else:
            exceeded_buffer_length = self.curr_iter > self.iter_to_save[-1]
            if exceeded_buffer_length:
                self.iter_to_save = [2 * i for i in self.iter_to_save]

            if self.curr_iter in self.iter_to_save:
                buffer_has_empty_slot = None in self.saved_iter
                if buffer_has_empty_slot:
                    index = self.saved_iter.index(None)
                else:
                    unneeded_entries = [
                        i
                        for i in self.saved_iter
                        if (i is not None and i not in self.iter_to_save)
                    ]
                    oldest_unneeded_entry = min(unneeded_entries)
                    index = self.saved_iter.index(oldest_unneeded_entry)

                self.saved_values[index] = copy.deepcopy(value)
                self.last = self.saved_values[index]
                self.saved_iter[index] = self.curr_iter
            else:
                self.last = copy.deepcopy(value)
        self.curr_iter += 1

    def get(self):
        iterations = [0] + list(filter(lambda x: x is not None, self.saved_iter))
        data = [self.first] + list(filter(lambda x: x is not None, self.saved_values))

        last_element_already_in_list = (self.curr_iter - 1) in self.saved_iter
        if not last_element_already_in_list and self.last is not None:
            iterations += [self.curr_iter - 1]
            data += [self.last]

        sort_index = np.argsort(iterations)

        iterations = [iterations[i] for i in sort_index]
        data = [data[i] for i in sort_index]

        return iterations, data
