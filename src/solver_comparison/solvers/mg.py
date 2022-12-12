import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray
from scipy import optimize

from solver_comparison.problem.model import Model
from solver_comparison.solvers.optimizer import CallbackFunction, Optimizer


@dataclass
class MG(Optimizer):
    """The Multiplicative Gradient algorithm."""

    tol: float = 10**-20

    def run(
        self, model: Model, param: NDArray, callback: Optional[CallbackFunction] = None
    ) -> NDArray:

        curr_param = param
        for t in range(self.max_iter):

            curr_grad = model.logp_grad(curr_param)[1]
            new_param = curr_param * curr_grad
            new_param = new_param / np.sum(new_param)

            if callback is not None:
                callback(new_param, None)

            if np.isnan(new_param).any():
                warnings.warn(
                    "iterates have a NaN a iteration {iter}; returning previous iterate"
                )

            curr_param = new_param

        return curr_param
