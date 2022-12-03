from functools import partial

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import softmax

from solver_comparison.problem.model import SIMPLEX, SOFTMAX, Model
from solver_comparison.problem.problem import Problem
from solver_comparison.solvers.expgrad import ExpGrad
from solver_comparison.solvers.frank_wolfe import AwayFrankWolfe, FrankWolfe
from solver_comparison.solvers.initializer import Initializer, InitUniform
from solver_comparison.solvers.lbfgs import LBFGS


def normalize(weights):
    _weights = np.array(weights).astype(float)
    return _weights / np.sum(_weights)


def gradient_finite_differences(func, param):
    n_dim = np.shape(param)[0]
    delta = 2 * np.sqrt(1e-12) * (1 + np.linalg.norm(param))
    grad = np.zeros(n_dim)
    e_i = np.zeros(n_dim)
    for i in range(n_dim):
        e_i[i] = 1
        fxp = func(param + delta * e_i)
        fxm = func(param - delta * e_i)
        grad[i] = (fxp - fxm) / (2 * delta)
        e_i[i] = 0
    return grad


class ToyModel(Model):
    def __init__(
        self,
        data: NDArray,
        use_softmax,
    ):
        self.data = data
        self.use_softmax = use_softmax

    def probabilities(self, w):
        return softmax(w) if self.use_softmax else w

    def objfunc_along_direction(self, param, direction):
        def _f(ss: float):
            return self.logp_grad(param + ss * direction)[0]

        return _f

    def logp_grad(self, theta=None, nograd=False, Hessinv=None, *args, **kwargs):
        n_samples = self.data.shape[0]

        def func(param):
            probs = softmax(param) if self.use_softmax else param
            return np.mean(np.log(self.data @ probs))

        def grad(param):
            probs = softmax(param) if self.use_softmax else param
            x_times_p = self.data @ probs
            _grad = np.einsum("nd,n->d", self.data, 1 / x_times_p) / n_samples

            if self.use_softmax:
                probs_vec = probs.reshape((-1, 1))
                correction = np.diag(probs) - probs_vec @ probs_vec.T
                _grad = correction @ _grad

            return _grad

        if nograd:
            return func(theta)
        else:
            return func(theta), grad(theta)


@pytest.mark.parametrize("use_softmax", [False, True])
def test_gradients(use_softmax):
    toy_model = ToyModel(data_uniform, use_softmax=use_softmax)

    _, grad_val = toy_model.logp_grad(w_shifted)

    func = partial(toy_model.logp_grad, nograd=True)
    grad_numerical = gradient_finite_differences(func, w_shifted)

    assert np.allclose(grad_val, grad_numerical)


data_uniform = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
).astype(float)
data_shifted = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
    ]
).astype(float)

w_uniform = normalize([1, 1, 1])
w_shifted = normalize([1, 1, 3])
w_unbalanced = normalize([1, 100, 100])


@pytest.mark.parametrize(
    "optimizer,data,param_start,expected",
    [
        (ExpGrad(), data_uniform, w_uniform, w_uniform),
        (ExpGrad(), data_uniform, w_shifted, w_uniform),
        (ExpGrad(), data_uniform, w_unbalanced, w_uniform),
        (ExpGrad(), data_shifted, w_uniform, w_shifted),
        (ExpGrad(), data_shifted, w_shifted, w_shifted),
        (ExpGrad(), data_shifted, w_unbalanced, w_shifted),
        (LBFGS(), data_uniform, w_uniform, w_uniform),
        (LBFGS(), data_uniform, w_shifted, w_uniform),
        (LBFGS(), data_uniform, w_unbalanced, w_uniform),
        (LBFGS(), data_shifted, w_uniform, w_shifted),
        (LBFGS(), data_shifted, w_shifted, w_shifted),
        (LBFGS(), data_shifted, w_unbalanced, w_shifted),
        (FrankWolfe(ls_ss_tol=1e-10), data_uniform, w_uniform, w_uniform),
        (FrankWolfe(ls_ss_tol=1e-10), data_uniform, w_shifted, w_uniform),
        (FrankWolfe(ls_ss_tol=1e-10), data_uniform, w_unbalanced, w_uniform),
        (FrankWolfe(ls_ss_tol=1e-10), data_shifted, w_uniform, w_shifted),
        (FrankWolfe(ls_ss_tol=1e-10), data_shifted, w_shifted, w_shifted),
        (FrankWolfe(ls_ss_tol=1e-10), data_shifted, w_unbalanced, w_shifted),
        (AwayFrankWolfe(ls_ss_tol=1e-20), data_uniform, w_uniform, w_uniform),
        (AwayFrankWolfe(ls_ss_tol=1e-20), data_uniform, w_shifted, w_uniform),
        (AwayFrankWolfe(ls_ss_tol=1e-20), data_uniform, w_unbalanced, w_uniform),
        (AwayFrankWolfe(ls_ss_tol=1e-20), data_shifted, w_uniform, w_shifted),
        (AwayFrankWolfe(ls_ss_tol=1e-20), data_shifted, w_shifted, w_shifted),
        (AwayFrankWolfe(ls_ss_tol=1e-20), data_shifted, w_unbalanced, w_shifted),
    ],
)
def test_optimizers(
    optimizer,
    data,
    param_start,
    expected,
):
    np.seterr(all="raise")
    use_softmax = isinstance(optimizer, LBFGS)
    toy_model = ToyModel(data, use_softmax=use_softmax)
    param_end = optimizer.run(toy_model, param_start)
    assert np.allclose(toy_model.probabilities(param_end), expected)


K, N, L = 8, 100, 14


@pytest.mark.parametrize(
    "problem",
    [
        Problem(
            model_type=SIMPLEX, filename="test5.fsa", K=K, N=N, L=L, alpha=0.1, beta=1.0
        ),
        Problem(
            model_type=SOFTMAX, filename="test5.fsa", K=K, N=N, L=L, alpha=0.1, beta=1.0
        ),
    ],
)
def test_gradient_rewrite(problem):
    model = problem.load_model()

    w0 = InitUniform().initialize_model(model)

    f_new, g_new = model._objective.func_and_grad(w0)
    f_old, g_old = model.kmerexpr_model.logp_grad(theta=w0)
    assert f_new == f_old
    assert np.allclose(g_new, g_old)

    w0 = normalize(w0 + np.random.randn(np.shape(w0)[0]) ** 2)

    f_new, g_new = model._objective.func_and_grad(w0)
    f_old, g_old = model.kmerexpr_model.logp_grad(theta=w0)
    assert f_new == f_old
    assert np.allclose(g_new, g_old)
