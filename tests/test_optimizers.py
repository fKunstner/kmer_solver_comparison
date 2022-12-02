from functools import partial
from typing import Literal, Union

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import softmax

from solver_comparison.problem.model import Simplex, SimplexModel, Softmax
from solver_comparison.problem.snapshot import Snapshot
from solver_comparison.solvers.expgrad import ExpGrad
from solver_comparison.solvers.lbfgs import LBFGS


def normalize(ws):
    w = np.array(ws).astype(float)
    return w / np.sum(w)


class ToyModel(SimplexModel):
    def __init__(
        self,
        X: NDArray,
        use_softmax,
    ):
        self.X = X
        self.use_softmax = use_softmax

    def probabilities(self, w):
        return softmax(w) if self.use_softmax else w

    def logp_grad(self, theta=None, nograd=False, Hessinv=None, *args, **kwargs):
        n, d = self.X.shape

        def f(w):
            p = softmax(w) if self.use_softmax else w
            return np.mean(np.log(self.X @ p))

        def g(w):
            p = softmax(w) if self.use_softmax else w
            x_times_p = self.X @ p
            grad = np.einsum("nd,n->d", self.X, 1 / x_times_p) / n

            if self.use_softmax:
                pvec = p.reshape((-1, 1))
                correction = np.diag(p) - pvec @ pvec.T
                grad = correction @ grad

            return grad

        if nograd:
            return f(theta)
        else:
            return f(theta), g(theta)


@pytest.mark.parametrize("use_softmax", [False, True])
def test_gradients(use_softmax):
    toy_model = ToyModel(Xuniform, use_softmax=use_softmax)

    def gradient_finite_differences(func, x):
        n = np.shape(x)[0]
        delta = 2 * np.sqrt(1e-12) * (1 + np.linalg.norm(x))
        g = np.zeros(n)
        e_i = np.zeros(n)
        for i in range(n):
            e_i[i] = 1
            fxp = func(x + delta * e_i)
            fxm = func(x - delta * e_i)
            g[i] = (fxp - fxm) / (2 * delta)
            e_i[i] = 0
        return g

    _, g = toy_model.logp_grad(w0_shifted)
    func = partial(toy_model.logp_grad, nograd=True)
    g_num = gradient_finite_differences(func, w0_shifted)

    assert np.allclose(g, g_num)


Xuniform = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
).astype(float)
Xshifted = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
    ]
).astype(float)

w0_uniform = normalize([1, 1, 1])
w0_shifted = normalize([1, 1, 3])
w0_very_shifted = normalize([1, 100, 100])


@pytest.mark.parametrize(
    "optimizer,X,start,expected",
    [
        (ExpGrad(), Xuniform, w0_uniform, w0_uniform),
        (ExpGrad(), Xuniform, w0_shifted, w0_uniform),
        (ExpGrad(), Xuniform, w0_very_shifted, w0_uniform),
        (ExpGrad(), Xshifted, w0_uniform, w0_shifted),
        (ExpGrad(), Xshifted, w0_shifted, w0_shifted),
        (ExpGrad(), Xshifted, w0_very_shifted, w0_shifted),
        (LBFGS(), Xuniform, w0_uniform, w0_uniform),
        (LBFGS(), Xuniform, w0_shifted, w0_uniform),
        (LBFGS(), Xuniform, w0_very_shifted, w0_uniform),
        (LBFGS(), Xshifted, w0_uniform, w0_shifted),
        (LBFGS(), Xshifted, w0_shifted, w0_shifted),
        (LBFGS(), Xshifted, w0_very_shifted, w0_shifted),
    ],
)
def test_simplex_models(
    optimizer,
    X,
    start,
    expected,
):
    np.seterr(all="raise")
    use_softmax = isinstance(optimizer, LBFGS)
    toymodel = ToyModel(X, use_softmax=use_softmax)
    snapshot = Snapshot(toymodel, start)
    out_snapshot = optimizer.run(snapshot)
    assert np.allclose(toymodel.probabilities(out_snapshot.param), expected)
