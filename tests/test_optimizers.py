from functools import partial

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import softmax

from solver_comparison.problem.model import SIMPLEX, SOFTMAX, Model
from solver_comparison.solvers.expgrad import ExpGrad
from solver_comparison.solvers.lbfgs import LBFGS


def normalize(weights):
    _weights = np.array(weights).astype(float)
    return _weights / np.sum(_weights)


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

    _, grad_val = toy_model.logp_grad(weights_shifted)

    func = partial(toy_model.logp_grad, nograd=True)
    grad_numerical = gradient_finite_differences(func, weights_shifted)

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

weights_uniform = normalize([1, 1, 1])
weights_shifted = normalize([1, 1, 3])
weights_unbalanced = normalize([1, 100, 100])


@pytest.mark.parametrize(
    "optimizer,data,param_start,expected",
    [
        (ExpGrad(), data_uniform, weights_uniform, weights_uniform),
        (ExpGrad(), data_uniform, weights_shifted, weights_uniform),
        (ExpGrad(), data_uniform, weights_unbalanced, weights_uniform),
        (ExpGrad(), data_shifted, weights_uniform, weights_shifted),
        (ExpGrad(), data_shifted, weights_shifted, weights_shifted),
        (ExpGrad(), data_shifted, weights_unbalanced, weights_shifted),
        (LBFGS(), data_uniform, weights_uniform, weights_uniform),
        (LBFGS(), data_uniform, weights_shifted, weights_uniform),
        (LBFGS(), data_uniform, weights_unbalanced, weights_uniform),
        (LBFGS(), data_shifted, weights_uniform, weights_shifted),
        (LBFGS(), data_shifted, weights_shifted, weights_shifted),
        (LBFGS(), data_shifted, weights_unbalanced, weights_shifted),
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
