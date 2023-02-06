"""Goal: Show that the convergence criterions can detect when we're done"""


from solver_comparison.exp_defs import make_experiment_for_opts
from solver_comparison.plotting import make_convergence_criterion_plots
from solver_comparison.solvers.expgrad import ExpGrad
from solver_comparison.solvers.lbfgs import LBFGS
from solver_comparison.solvers.mg import MG

filename = "sampled_genome_0.1.fsa"
max_iter = 1_000
K, N, L, alpha = 14, 5_000_000, 100, 0.1


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
