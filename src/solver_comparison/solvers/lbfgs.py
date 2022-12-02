import warnings
from dataclasses import dataclass
from typing import Optional

from numpy.typing import NDArray
from scipy import optimize

from solver_comparison.problem.model import Model
from solver_comparison.solvers.optimizer import CallbackFunction, Optimizer


@dataclass
class LBFGS(Optimizer):
    """LBFGS using the Scipy implementation."""

    def run(
        self,
        model: Model,
        param: NDArray,
        progress_callback: Optional[CallbackFunction] = None,
    ) -> NDArray:
        def lbfgs_callback(_param: NDArray):
            if progress_callback is not None:
                progress_callback(_param, None)

        def func(theta):
            return -model.logp_grad(theta)[0]

        def grad(theta):
            return -model.logp_grad(theta)[1]

        final_param, f_sol, dict_flags_convergence = optimize.fmin_l_bfgs_b(
            func,
            param,
            grad,
            pgtol=1e-12,
            factr=1.0,
            maxiter=self.max_iter,
            maxfun=10 * self.max_iter,
            callback=lbfgs_callback,
        )

        if dict_flags_convergence["warnflag"] == 1:
            warnings.warn(
                "softmax model did not converge. "
                "Too many function evaluations or too many iterations. "
                f"Total iterations: {dict_flags_convergence['nit']}"
                f"Print d[task]: {dict_flags_convergence['task']}"
            )
        elif dict_flags_convergence["warnflag"] == 2:
            warnings.warn(
                f"Softmax model did not converge due to: {dict_flags_convergence['task']}"
            )

        return final_param
