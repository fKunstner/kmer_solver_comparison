from functools import partial

from solver_comparison.experiment import Experiment
from solver_comparison.plotting import (
    make_scatter_comparison_plots,
    make_test_error_comparison_plots,
)
from solver_comparison.problem.model import SIMPLEX, SOFTMAX
from solver_comparison.problem.problem import Problem
from solver_comparison.solvers.expgrad import ExpGrad
from solver_comparison.solvers.frank_wolfe import AwayFrankWolfe, FrankWolfe
from solver_comparison.solvers.initializer import Initializer, InitUniform
from solver_comparison.solvers.lbfgs import LBFGS
from solver_comparison.solvers.mg import MG

problem = "small"

if problem == "small":
    filename = "test5.fsa"
    K, N, L, alpha = 8, 100, 14, 0.1
    max_iter = 100
elif problem == "medium":
    filename = "sampled_genome_0.01.fsa"
    max_iter = 1000
    K, N, L, alpha = 14, 500000, 50, 0.1
else:
    raise ValueError(f"Problem {problem} unknown")

SoftmaxProblem = partial(Problem, model_type=SOFTMAX)
SimplexProblem = partial(Problem, model_type=SIMPLEX)


experiment = Experiment(
    prob=SimplexProblem(filename=filename, K=K, N=N, L=L, alpha=alpha, beta=1.0),
    opt=MG(max_iter=max_iter),
    init=InitUniform(),
)


if __name__ == "__main__":

    experiment.run()

    make_scatter_comparison_plots([experiment])
    make_test_error_comparison_plots([experiment])
