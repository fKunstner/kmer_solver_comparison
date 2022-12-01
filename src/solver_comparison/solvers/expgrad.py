from dataclasses import dataclass
from typing import ClassVar

from kmerexpr.exp_grad_solver import exp_grad_solver
from solver_comparison.logging.sequence_summarizer import OnlineSequenceSummary
from solver_comparison.problem.snapshot import Snapshot
from solver_comparison.solvers.optimizer import CallbackFunction, Optimizer


@dataclass
class ExpGrad(Optimizer):
    """Exponentiated Gradient Descent / Mirror Descent with a line search.

    Calls exp_grad_solver with the default Armijo linesearch. Does not use
    additional features (HessInv, lrs)

    Args:
        verbose: Enables print statements
    """

    max_iter: int = 1000
    verbose: bool = True
    solver_name: ClassVar[str] = "exp_grad"

    def run(self, curr_p: Snapshot, progress_callback: CallbackFunction) -> Snapshot:

        model, param = curr_p.model, curr_p.param

        dict_sol = exp_grad_solver(
            loss_grad=model.logp_grad,
            x_0=param,
            lrs="armijo",
            tol=10 ** (-8.0),
            gtol=10 ** (-8.0),
            n_iters=self.max_iter,
            verbose=self.verbose,
            n=None,
            Hessinv=False,
        )

        xs_summary = OnlineSequenceSummary(n=20)
        for x in dict_sol["xs"]:
            xs_summary.update(x)

        return Snapshot(model, dict_sol["x"])
