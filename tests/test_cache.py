import pytest

from solver_comparison.logging.utils import runtime
from solver_comparison.problem.model import SIMPLEX, SOFTMAX
from solver_comparison.problem.problem import Problem
from solver_comparison.solvers.initializer import Initializer, InitUniform

K, N, L = 8, 20, 14
problem_softmax = Problem(
    model_type=SOFTMAX, filename="test5.fsa", K=K, N=N, L=L, alpha=0.1, beta=1.0
)
problem_simplex = Problem(
    model_type=SIMPLEX, filename="test5.fsa", K=K, N=N, L=L, alpha=0.1, beta=1.0
)


@pytest.mark.parametrize(
    "problem, nograd",
    [
        (problem_simplex, True),
        (problem_simplex, False),
        (problem_softmax, True),
        (problem_softmax, False),
    ],
)
def test_timing_cache(problem, nograd):
    model = problem.load_model()
    param = InitUniform().initialize_model(model)

    with runtime() as first_call:
        model.logp_grad(param, nograd=nograd)

    with runtime() as second_call:
        model.logp_grad(param, nograd=nograd)

    assert second_call.time < first_call.time


@pytest.mark.parametrize("problem", [problem_simplex, problem_softmax])
def test_timing_cache_function_after_grad(problem):
    model = problem.load_model()
    param = InitUniform().initialize_model(model)

    with runtime() as first_call:
        model.logp_grad(param, nograd=False)

    with runtime() as second_call:
        model.logp_grad(param, nograd=True)

    assert second_call.time < 0.1 * first_call.time
