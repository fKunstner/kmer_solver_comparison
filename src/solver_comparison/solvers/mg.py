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

    tol: float = 10**-6

    def fw_gap(self, param, grad):
        """The Frank-Wolfe gap an upper bound on the optimality gap.

        the loss f (negative of the objectve) is convex,
            f(y) >= f(x) + <f'(x), y - x>
        We can thus bound the optimality gap by
            f(x) - f(x*) <= - min_y <f'(x), y - x> : y in simplex
        and the RHS is minimized
        """
        # For a convex f, would be
        # -(np.min(grad) - np.inner(grad, param))
        # In our case, grad is the negative of the gradient, so
        # -(np.min(-grad) - np.inner(-grad, param))
        # simplifies to
        return np.max(grad) - np.inner(grad, param)

    def run(
        self, model: Model, param: NDArray, callback: Optional[CallbackFunction] = None
    ) -> NDArray:

        curr_param = param
        for t in range(self.max_iter):

            curr_obj, curr_grad = model.logp_grad(curr_param)
            new_param = curr_param * curr_grad
            new_param = new_param
            new_obj, new_grad = model.logp_grad(new_param)

            if callback is not None:
                callback(new_param, None)

            if np.isnan(new_param).any():
                warnings.warn(
                    "iterates have a NaN a iteration {iter}; returning previous iterate"
                )

            if self.fw_gap(new_param, new_grad) < self.tol:
                print(f"Converged within tolerance")
                break

            curr_param = new_param

        return curr_param
