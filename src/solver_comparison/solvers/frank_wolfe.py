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
class FrankWolfe(Optimizer):
    """Frank-Wolfe optimizer."""

    tol: float = 10**-20
    ls_ss_tol: float = 1e-5

    def linesearch(self, objective_ss):
        result = optimize.minimize_scalar(
            lambda ss: -objective_ss(ss),
            bounds=(0.0, 1.0),
            method="Bounded",
            options={"xatol": self.ls_ss_tol},
        )
        return result["x"]

    def step(self, model: Model, curr_param: NDArray) -> Tuple[NDArray, float]:
        """A Frank-Wolfe step on the simplex."""

        curr_grad = model.logp_grad(curr_param)[1]

        imax = np.argmax(curr_grad)
        param_target = np.zeros_like(curr_param)
        param_target[imax] = 1.0
        direction = param_target - curr_param

        primal_dual_gap = curr_grad.dot(direction)

        objective_ss = model.objfunc_along_direction(curr_param, direction)
        stepsize = self.linesearch(objective_ss)
        curr_param = curr_param + stepsize * direction
        return curr_param, primal_dual_gap

    def run(
        self, model: Model, param: NDArray, callback: Optional[CallbackFunction] = None
    ) -> NDArray:

        curr_param = param
        for t in range(self.max_iter):
            new_param, primal_dual_gap = self.step(model, curr_param)

            if callback is not None:
                callback(new_param, None)

            import pdb

            if np.isnan(new_param).any():
                warnings.warn(
                    "iterates have a NaN a iteration {iter}; returning previous iterate"
                )
                pdb.set_trace()
            if primal_dual_gap < self.tol:
                print(f"Optimality gap is less than: {self.tol}, stopping.")
                break

            curr_param = new_param

        return curr_param


@dataclass
class AwayFrankWolfe(FrankWolfe):
    """Frank-Wolfe with away steps."""

    def step(self, model: Model, curr_param: NDArray) -> Tuple[NDArray, float]:
        curr_grad: NDArray = model.logp_grad(curr_param)[1]
        target: NDArray = np.zeros_like(curr_param)
        away_target: NDArray = np.zeros_like(curr_param)

        imax = np.argmax(curr_grad)
        target[imax] = 1.0
        incr_direction = target - curr_param
        incr_projection = curr_grad.dot(incr_direction)

        primal_dual_gap = incr_projection

        valid_directions = curr_grad.copy()
        valid_directions[curr_param == 0] = np.inf
        imin = np.argmin(valid_directions)
        away_target[imin] = 1.0
        decr_direction = curr_param - away_target
        decr_projection = curr_grad.dot(decr_direction)

        if incr_projection > decr_projection:
            direction = incr_direction
            max_ss = 1.0
        else:
            direction = decr_direction
            max_ss = -curr_param[imin] / decr_direction[imin]

        objective_ss = model.objfunc_along_direction(curr_param, direction)

        def rescaled_objective(stepsize_in_0_1):
            return objective_ss(stepsize_in_0_1 * max_ss)

        stepsize = self.linesearch(rescaled_objective)

        stepsize_is_close_to_1 = np.abs(stepsize - 1.0) < 10 * self.ls_ss_tol
        if stepsize_is_close_to_1:
            should_be_1 = rescaled_objective(1.0) >= rescaled_objective(stepsize)
            if should_be_1:
                stepsize = 1.0

        curr_param = curr_param + stepsize * max_ss * direction
        return curr_param, primal_dual_gap
