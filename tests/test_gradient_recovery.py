import numpy as np
from scipy.special import softmax

from solver_comparison.plotting.data import grad_softmax_to_grad_simplex
from solver_comparison.problem.model import SIMPLEX, SOFTMAX
from solver_comparison.problem.problem import Problem
from solver_comparison.solvers.initializer import InitUniform


def test_gradient_rewrite():
    K, N, L = 8, 100, 14

    simplex_model = Problem(
        model_type=SIMPLEX, filename="test5.fsa", K=K, N=N, L=L, alpha=0.1, beta=1.0
    ).load_model()
    softmax_model = Problem(
        model_type=SOFTMAX, filename="test5.fsa", K=K, N=N, L=L, alpha=0.1, beta=0.0
    ).load_model()

    def check_at(weights):
        _, grad_softmax = softmax_model.logp_grad(param=weights)
        _, grad_simplex = simplex_model.logp_grad(param=softmax(weights))
        print("Should be ", grad_simplex)
        recovered_grad_simplex = grad_softmax_to_grad_simplex(weights, grad_softmax)
        print("Got       ", recovered_grad_simplex)
        print(grad_simplex, recovered_grad_simplex)
        assert np.allclose(recovered_grad_simplex, grad_simplex)

    weights = InitUniform().initialize_model(softmax_model)
    check_at(weights)

    check_at(np.random.randn(*weights.shape))
