from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from solver_comparison.problem.model import Model
from solver_comparison.solvers.frank_wolfe import FrankWolfe


@dataclass
class MyAwayFrankWolfe(FrankWolfe):
    """Frank-Wolfe with away steps.

    Failed attempt
    """

    def step(self, model: Model, curr_param: NDArray) -> Tuple[NDArray, float]:
        curr_grad: NDArray = model.logp_grad(curr_param)[1]
        target: NDArray = np.zeros_like(curr_param)
        away_target: NDArray = np.zeros_like(curr_param)

        imax = np.argmax(curr_grad)
        target[imax] = 1.0
        incr_direction = target - curr_param
        incr_projection = curr_grad.dot(incr_direction)

        max_ss_per_coord = curr_param / (1 - curr_param)
        grad_param_inner_product = curr_grad.dot(curr_param)
        progres_per_direction = (
            max_ss_per_coord * grad_param_inner_product - max_ss_per_coord * curr_grad
        )
        jmax = np.argmax(progres_per_direction)
        away_target[jmax] = 1.0
        decr_direction = max_ss_per_coord[jmax] * (curr_param - away_target)
        decr_projection = curr_grad.dot(decr_direction)

        primal_dual_gap = incr_projection
        if incr_projection > decr_projection:
            direction = incr_direction
        else:
            print("AWAY")
            direction = decr_direction

        objective_ss = model.objfunc_along_direction(curr_param, direction)

        stepsize = self.linesearch(objective_ss)
        curr_param = curr_param + stepsize * direction
        print(
            " ".join([f"{x:.4e}" for x in list(curr_param)]),
            # " ".join([f"{x:.4e}" for x in list(curr_grad)]),
        )
        return curr_param, primal_dual_gap
