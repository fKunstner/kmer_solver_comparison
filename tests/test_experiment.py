import pytest

from solver_comparison.experiment import Experiment
from solver_comparison.plotting import make_individual_exp_plots
from solver_comparison.problem.problem import Problem
from solver_comparison.solvers.expgrad import ExpGrad
from solver_comparison.solvers.initializer import Initializer
from solver_comparison.solvers.lbfgs import LBFGS

K, N, L = 8, 100, 14
problem_logistic = Problem(
    model_name="Logistic", filename="test5.fsa", K=K, N=N, L=L, alpha=0.1, beta=1.0
)
problem_simplex = Problem(
    model_name="Simplex", filename="test5.fsa", K=K, N=N, L=L, alpha=0.1, beta=1.0
)


@pytest.mark.parametrize(
    "optimizer,problem",
    [(ExpGrad(), problem_simplex), (LBFGS(), problem_logistic)],
)
def test_simplex_models(optimizer, problem):
    Experiment(
        prob=problem,
        opt=optimizer,
        init=Initializer("simplex_uniform"),
    ).run()
