from dataclasses import dataclass
from typing import ClassVar, Tuple

from numpy.typing import NDArray
from scipy import optimize
from scipy.special import softmax
from solver_comparison.logging.sequence_summarizer import OnlineSequenceSummary
from solver_comparison.problem.snapshot import Snapshot
from solver_comparison.solvers.optimizer import CallbackFunction, Optimizer


@dataclass
class LBFGS(Optimizer):
    """LBFGS using the Scipy implementation."""

    solver_name: ClassVar[str] = "lbfgs"

    def step(self, current: Snapshot) -> Snapshot:
        raise ValueError

    def run(
        self,
        curr_p: Snapshot,
        progress_callback: CallbackFunction,
    ) -> Tuple[Snapshot, int, OnlineSequenceSummary]:

        model, param = curr_p.model, curr_p.param

        xs_summary = OnlineSequenceSummary(n=20)

        lbfgs_iter_counter = 0

        def lbfgs_callback(x: NDArray):
            nonlocal lbfgs_iter_counter
            lbfgs_iter_counter += 1
            progress_callback(
                Snapshot(model, x), lbfgs_iter_counter, self.max_iter, None
            )
            xs_summary.update(softmax(x))

        def func(theta):
            return -model.logp_grad(theta)[0]

        def grad(theta):
            return -model.logp_grad(theta)[1]

        theta_sol, f_sol, dict_flags_convergence = optimize.fmin_l_bfgs_b(
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
            print(
                "WARNING: softmax model did not converge. "
                "Too many function evaluations or too many iterations. "
                "Print d[task]:",
                dict_flags_convergence["task"],
            )
            print("Total iterations: ", str(dict_flags_convergence["nit"]))
        elif dict_flags_convergence["warnflag"] == 2:
            print(
                "WARNING: softmax model did not converge due to: ",
                dict_flags_convergence["task"],
            )

        return Snapshot(model, theta_sol), dict_flags_convergence["nit"], xs_summary
