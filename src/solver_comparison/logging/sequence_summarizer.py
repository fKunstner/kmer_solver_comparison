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
        self.saved = [None] * n_to_save
        self.it_saved = [None] * n_to_save
        self.it_to_save = [i + 1 for i in range(n_to_save)]
        self.iteration = 0

    def update(self, value):
        if self.iteration == 0:
            self.first = deepcopy(value)

        else:
            if self.iteration > self.it_to_save[-1]:
                self.it_to_save = [2 * i for i in self.it_to_save]

            if self.iteration in self.it_to_save:
                if None in self.it_saved:
                    index = self.it_saved.index(None)
                else:
                    iterations_no_longer_needed = [
                        i
                        for i in self.it_saved
                        if (i is not None and i not in self.it_to_save)
                    ]
                    iteration_to_discard = min(iterations_no_longer_needed)
                    index = self.it_saved.index(iteration_to_discard)

                self.saved[index] = copy.deepcopy(value)
                self.last = self.saved[index]
                self.it_saved[index] = self.iteration
            else:
                self.last = copy.deepcopy(value)
        self.iteration += 1

    def get(self):
        iterations = [0] + [_ for _ in self.it_saved if _ is not None]
        data = [self.first] + [_ for _ in self.saved if _ is not None]

        last_element_already_in_list = (self.iteration - 1) in self.it_saved
        if not last_element_already_in_list and self.last is not None:
            iterations += [self.iteration - 1]
            data += [self.last]

        sort_index = np.argsort(iterations)

        iterations = [iterations[i] for i in sort_index]
        data = [data[i] for i in sort_index]

        return iterations, data
