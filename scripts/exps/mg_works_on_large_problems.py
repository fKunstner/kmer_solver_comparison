"""Goal: Show MG gives good solutions fast for large problems"""

from solver_comparison.exp_defs import make_experiment_for_opts
from solver_comparison.plotting import (
    make_convergence_criterion_plots,
    make_scatter_comparison_plots,
    make_test_error_comparison_plots,
)
from solver_comparison.problem import problem_settings
from solver_comparison.solvers.expgrad import ExpGrad
from solver_comparison.solvers.lbfgs import LBFGS
from solver_comparison.solvers.mg import MG

max_iter = 1_000

experiments_per_dataset = [
    make_experiment_for_opts(
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
    for filename, K, N, L, alpha in [
        problem_settings["massive-sparse"],
        problem_settings["massive-medium"],
        problem_settings["massive-dense"],
    ]
]


if __name__ == "__main__":
    #    for experiments in experiments_per_dataset:
    #        for exp in experiments:
    #            print(exp.as_dict())
    #            if exp.has_already_run():
    #                print("stored at ", exp.hash())
    #            else:
    #                exp.run()

    for experiments in experiments_per_dataset:
        make_convergence_criterion_plots(experiments)
        make_test_error_comparison_plots(experiments)

    for experiments in experiments_per_dataset:
        make_scatter_comparison_plots(experiments)

    for experiments in experiments_per_dataset:
        for exp in experiments:
            make_scatter_comparison_plots([exp])
