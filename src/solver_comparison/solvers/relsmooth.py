"""Implementation of the relative smoothness solver of Bauschke et al.

Heinz H. Bauschke, Jérôme Bolte, Marc Teboulle (2016) A Descent Lemma
Beyond Lipschitz Gradient Continuity: First-Order Methods Revisited and
Applications. Mathematics of Operations Research 42(2):330-348.

Adapted from the algorithm for Poisson Linear Inverse Problems in Section 5.2
"""
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from solver_comparison.problem.model import Model
from solver_comparison.solvers.optimizer import CallbackFunction, Optimizer


@dataclass
class RelSmooth(Optimizer):
    """Mirror descent with the Burg entropy.

    Uses the Burg entropy as the reference function `h(x) = -sum_i log(x_i)`

    The original problem works for non-negative instead of simplex constraints.
    Instead of solving f(x) subject to simplex constraints, we can solve
    f(x) - |x|_1 and normalize the resulting x to obtain a probability.

    This works because the problem
        x* = arg min_x f(x) - |x|_1
            subject to x >= 0
    is equivalent to the following on the set |x|_1 = |x*|_1
        x* = arg min_x f(x)
            subject to x >= 0 and |x|_1 = |x*|_1.
    We can then normalize the inputs to f as f(c x) = f(x) + log(c) to get
        x*/|x*|_1 = arg min_x f(x/|x*|_1) - log(|x*|_1)
            subject to x >= 0 and |x|_1 = |x*|_1
    """

    max_iter: int = 1000

    def step(self, model: Model, curr_param: NDArray, stepsize: float) -> NDArray:
        grad_with_l1_reg = -model.logp_grad(curr_param)[1] + curr_param
        new_param = 1 / (stepsize * grad_with_l1_reg + 1 / curr_param)
        return new_param

    def run(
        self,
        model: Model,
        param: NDArray,
        progress_callback: Optional[CallbackFunction] = None,
    ) -> NDArray:

        curr_param = param
        for t in range(self.max_iter):
            new_param = self.step(model, curr_param, stepsize=1.0)

            if progress_callback is not None:
                progress_callback(new_param / np.sum(new_param), None)

            if np.isnan(new_param).any():
                warnings.warn(
                    "iterates have a NaN a iteration {iter}; returning previous iterate"
                )

            curr_param = new_param

        return curr_param


@dataclass
class RelSmoothLS(RelSmooth):
    """RelSmooth algorithm with LineSearch."""

    def relative_armijo_decrease(
        self, old_param: NDArray, new_param: NDArray, stepsize: float
    ):
        def h(x):
            return -np.sum(np.log(x))

        def dh(x):
            return -1 / x

        def D(x, y):
            return h(x) - h(y) - np.inner(dh(y), x - y)

        return D(old_param, new_param) / stepsize

    def run(
        self,
        model: Model,
        param: NDArray,
        progress_callback: Optional[CallbackFunction] = None,
    ) -> NDArray:

        curr_param = param
        curr_f = model.logp_grad(curr_param, nograd=True)
        curr_ssize = 1.0
        for t in range(self.max_iter):

            while True:
                new_param = self.step(model, curr_param, stepsize=curr_ssize)
                new_f = model.logp_grad(new_param, nograd=True)
                decr = self.relative_armijo_decrease(curr_param, new_param, curr_ssize)
                if -new_f <= -curr_f - decr:
                    curr_ssize = curr_ssize * 2
                    break
                else:
                    curr_ssize = curr_ssize / 2

            if progress_callback is not None:
                progress_callback(new_param / np.sum(new_param), None)

            if np.isnan(new_param).any():
                warnings.warn(
                    "iterates have a NaN a iteration {iter}; returning previous iterate"
                )

            curr_param = new_param

        return curr_param
