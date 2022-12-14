import pytest

from solver_comparison.experiment import Experiment
from solver_comparison.problem.model import SIMPLEX, SOFTMAX
from solver_comparison.problem.problem import Problem
from solver_comparison.solvers.expgrad import ExpGrad
from solver_comparison.solvers.frank_wolfe import AwayFrankWolfe, FrankWolfe
from solver_comparison.solvers.initializer import Initializer, InitUniform
from solver_comparison.solvers.lbfgs import LBFGS

K, N, L = 8, 100, 14
problem_softmax = Problem(
    model_type=SOFTMAX, filename="test5.fsa", K=K, N=N, L=L, alpha=0.1, beta=1.0
)
problem_simplex = Problem(
    model_type=SIMPLEX, filename="test5.fsa", K=K, N=N, L=L, alpha=0.1, beta=1.0
)


@pytest.mark.parametrize(
    "optimizer,problem",
    [
        (ExpGrad(), problem_simplex),
        (FrankWolfe(), problem_simplex),
        (AwayFrankWolfe(), problem_simplex),
        (LBFGS(), problem_softmax),
    ],
)
def test_no_error_on_simple_experiment(optimizer, problem):
    Experiment(
        prob=problem,
        opt=optimizer,
        init=InitUniform(),
    ).run()
