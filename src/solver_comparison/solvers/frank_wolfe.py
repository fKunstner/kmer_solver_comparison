import time
import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import scipy as sp
from kmerexpr.exp_grad_solver import update_records
from numpy import maximum, sqrt
from numpy.linalg import norm
from numpy.typing import NDArray
from scipy import optimize
from scipy.special import softmax as softmax

from solver_comparison.problem.model import Model
from solver_comparison.solvers.optimizer import CallbackFunction, Optimizer


@dataclass
class FrankWolfe(Optimizer):
    """Frank-Wolfe optimizer."""

    away_step: bool = False
    pairwise_step: bool = False
    tol: float = 10**-20
    gtol: float = 10**-20

    def step(self, param: NDArray, loss: Callable, grad: NDArray) -> NDArray:
        imin = np.argmin(grad)
        param_target = np.zeros_like(param)
        param_target[imin] = 1.0
        direction = param_target - param

        def loss_update(stepsize):
            return loss(param + stepsize * direction)

        result = optimize.minimize_scalar(
            loss_update,
            bounds=(0.0, 1.0),
            method="Bounded",
            options={
                "disp": 3,
                "xatol": 1e-12,
            },
        )
        stepsize = result["x"]

        param = param + stepsize * direction

        return param

    def run(
        self,
        model: Model,
        param: NDArray,
        progress_callback: Optional[CallbackFunction] = None,
    ) -> NDArray:
        def loss(theta):
            return -model.logp_grad(theta)[0]

        def grad(theta):
            return -model.logp_grad(theta)[1]

        curr_param = param
        curr_grad = grad(curr_param)
        for t in range(self.max_iter):
            new_param = self.step(curr_param, loss, curr_grad)
            new_grad = grad(new_param)

            if progress_callback is not None:
                progress_callback(new_param)

            print(new_param)

            if np.isnan(new_param).any():
                warnings.warn(
                    "iterates have a NaN a iteration {iter}; returning previous iterate"
                )
                break
            #            if norm(new_param - curr_param, ord=1) <= self.tol:
            #                print(
            #                    f"Frank Wolfe iterates are less than: {self.tol}, apart. Stopping"
            #                )
            #                break
            #            if norm(new_grad - curr_grad, ord=1) <= self.gtol:
            #                print(f"Frank Wolfe grads are less than: {self.gtol}, apart. Stopping")
            #                break

            curr_grad = new_grad
            curr_param = new_param

        return curr_param
