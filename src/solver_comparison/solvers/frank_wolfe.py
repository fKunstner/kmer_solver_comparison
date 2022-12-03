import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray
from scipy import optimize

from solver_comparison.problem.model import Model
from solver_comparison.solvers.optimizer import CallbackFunction, Optimizer


@dataclass
class FrankWolfe(Optimizer):
    """Frank-Wolfe optimizer."""

    away_step: bool = False
    pairwise_step: bool = False
    tol: float = 10 ** -20
    gtol: float = 10 ** -20
    linesearch_stepsize_abs_tolerance: float = 1e-5

    def step(self, model: Model, curr_param: NDArray) -> NDArray:
        curr_grad = model.logp_grad(curr_param)[1]

        imax = np.argmax(curr_grad)
        param_target = np.zeros_like(curr_param)
        param_target[imax] = 1.0

        direction = param_target - curr_param

        objective_ss = model.objfunc_along_direction(curr_param, direction)

        result = optimize.minimize_scalar(
            lambda ss: -objective_ss(ss),
            bounds=(0.0, 1.0),
            method="Bounded",
            options={"xatol": self.linesearch_stepsize_abs_tolerance},
        )
        stepsize = result["x"]
        curr_param = curr_param + stepsize * direction

        return curr_param

    def run(
        self, model: Model, param: NDArray, callback: Optional[CallbackFunction] = None
    ) -> NDArray:

        curr_param = param
        curr_grad = model.logp_grad(curr_param)[1]
        for t in range(self.max_iter):
            new_param = self.step(model, curr_param)
            new_grad = model.logp_grad(new_param)[1]

            if callback is not None:
                callback(new_param, None)

            if np.isnan(new_param).any():
                warnings.warn(
                    "iterates have a NaN a iteration {iter}; returning previous iterate"
                )
                break
            if norm(new_param - curr_param, ord=1) <= self.tol:
                print(
                    f"Frank Wolfe iterates are less than: {self.tol}, apart. Stopping"
                )
                break
            if norm(new_grad - curr_grad, ord=1) <= self.gtol:
                print(f"Frank Wolfe grads are less than: {self.gtol}, apart. Stopping")
                break

            curr_param = new_param
            curr_grad = new_grad

        return curr_param
