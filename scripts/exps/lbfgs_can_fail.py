"""Goal: Highlight setting where LBFGS fails"""

from solver_comparison.exp_defs import make_experiment_for_opts
from solver_comparison.plotting import (
    make_convergence_criterion_plots,
    make_test_error_comparison_plots,
)
from solver_comparison.problem import problem_settings
from solver_comparison.solvers.expgrad import ExpGrad
from solver_comparison.solvers.lbfgs import LBFGS
from solver_comparison.solvers.mg import MG

max_iter = 1_000
filename, K, N, L, alpha = problem_settings["large-sparse"]


experiments = make_experiment_for_opts(
    filename,
    K,
    N,
    L,
    alpha,
    softmax_opts=[LBFGS(max_iter=max_iter)],
    simplex_opts=[
        ExpGrad(max_iter=max_iter),
        MG(max_iter=max_iter),
    ],
)


if __name__ == "__main__":
    for exp in experiments:
        print(exp.as_dict())
        if exp.has_already_run():
            print("stored at ", exp.hash())
        else:
            exp.run()

    make_convergence_criterion_plots(experiments)
    make_test_error_comparison_plots(experiments)
