from dataclasses import dataclass
from typing import Optional

from kmerexpr.exp_grad_solver import exp_grad_solver
from numpy.typing import NDArray

from solver_comparison.problem.model import Model
from solver_comparison.solvers.optimizer import CallbackFunction, Optimizer


@dataclass
class ExpGrad(Optimizer):
    """Exponentiated Gradient Ascent / Mirror Ascent with a line search.

    Calls exp_grad_solver with the default Armijo linesearch. Does not
    use additional features (HessInv, lrs)
    """

    max_iter: int = 1000

    def run(
        self,
        model: Model,
        param: NDArray,
        progress_callback: Optional[CallbackFunction] = None,
    ) -> NDArray:

        dict_sol = exp_grad_solver(
            # Todo: not a loss, an objective function (exp_grad maximizes)
            # Needs a change in kmerexpr
            loss_grad=model.logp_grad,
            x_0=param,
            lrs="armijo",
            tol=10 ** (-20.0),
            gtol=10 ** (-20.0),
            n_iters=self.max_iter,
            verbose=False,
            callback=progress_callback,
        )

        return dict_sol["x"]
