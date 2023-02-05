"""Main takeaway: Frank-Wolfe is slow.

Comparison of
- vanilla Frank-Wolfe (FW)
- away-step Frank-Wolfe (AFW)
- multiplicative gradient (MG)

Outputs an optimization plot showing that MG solves the problem
during the allocated iterations while (A)FW struggles to make progress.
This holds when looking at iteration or time performance.

Although (A)FW is supposed to be fast per-iteration, it still requires
computation of the full gradient (and needs a line-search on top of that,
or a finicky step-size schedule).

We probably shouldn't use it.
"""


from functools import partial

from solver_comparison.experiment import Experiment
from solver_comparison.plotting import make_optim_comparison_plots
from solver_comparison.problem.model import SIMPLEX, SOFTMAX
from solver_comparison.problem.problem import Problem
from solver_comparison.solvers.frank_wolfe import AwayFrankWolfe, FrankWolfe
from solver_comparison.solvers.initializer import InitUniform
from solver_comparison.solvers.mg import MG

filename = "sampled_genome_0.1.fsa"
max_iter = 100
K, N, L, alpha = 14, 5_000_000, 100, 0.1

SoftmaxProblem = partial(Problem, model_type=SOFTMAX)
SimplexProblem = partial(Problem, model_type=SIMPLEX)


experiments = [
    Experiment(
        prob=SimplexProblem(filename=filename, K=K, N=N, L=L, alpha=alpha, beta=1.0),
        opt=opt,
        init=InitUniform(),
    )
    for opt in [
        FrankWolfe(max_iter=max_iter),
        AwayFrankWolfe(max_iter=max_iter),
        MG(max_iter=max_iter),
    ]
]


if __name__ == "__main__":
    for exp in experiments:
        print(exp.as_dict())
        if exp.has_already_run():
            print("stored at ", exp.hash())
        else:
            exp.run()

    make_optim_comparison_plots(experiments)
