from typing import Literal

import numpy as np
import pytest
from scipy.special import softmax

from solver_comparison.problem.model import Simplex
from solver_comparison.problem.snapshot import Snapshot
from solver_comparison.solvers.expgrad import ExpGrad
from solver_comparison.solvers.lbfgs import LBFGS


def normalize(ws):
    w = np.array(ws).astype(float)
    return w / np.sum(w)


class ToyModel(Simplex):
    def __init__(
        self,
        X,
        direction: Literal["maximize", "minimize"] = "maximize",
        model: Literal["Simplex", "Logistic"] = "Simplex",
    ):
        self.X = X
        self.direction = direction
        self.model = model

    def probabilities(self, w):
        if self.model == "Simplex":
            return w
        elif self.model == "Logistic":
            return softmax(w)

    def logp_grad(self, theta=None, nograd=False, **kwargs):
        n, d = self.X.shape

        sign = -1 if self.direction == "minimize" else 1

        def f(w):
            if self.model == "Logistic":
                p = softmax(w)
            else:
                p = w
            f = sign * np.mean(np.log(self.X @ p))
            print(f)
            return f

        def g(w):
            if self.model == "Logistic":
                p = softmax(w)
            else:
                p = w
            print(w, p)

            Xp = self.X @ p
            grad = np.einsum("nd,n->d", self.X, 1 / Xp) / n
            if self.model == "Logistic":
                pvec = p.reshape((-1, 1))
                correction = np.diag(p) - pvec @ pvec.T
                grad = correction @ grad

            return sign * grad

        if nograd:
            return f(theta)
        else:
            return f(theta), g(theta)


@pytest.mark.parametrize("model", ["Simplex", "Logistic"])
def test_gradients(model: Literal["Simplex", "Logistic"]):
    toymodel = ToyModel(Xuniform, direction="maximize", model=model)

    f, g = toymodel.logp_grad(w0_shifted)

    def numGrad(func, x):
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

    gnum = numGrad(lambda w: toymodel.logp_grad(w, nograd=True), w0_shifted)

    assert np.allclose(g, gnum)


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
    "optimizer,model,direction,X,start,expected",
    [
        (ExpGrad(), "Simplex", "maximize", Xuniform, w0_uniform, w0_uniform),
        (ExpGrad(), "Simplex", "maximize", Xuniform, w0_shifted, w0_uniform),
        (ExpGrad(), "Simplex", "maximize", Xuniform, w0_very_shifted, w0_uniform),
        (ExpGrad(), "Simplex", "maximize", Xshifted, w0_uniform, w0_shifted),
        (ExpGrad(), "Simplex", "maximize", Xshifted, w0_shifted, w0_shifted),
        (ExpGrad(), "Simplex", "maximize", Xshifted, w0_very_shifted, w0_shifted),
        (LBFGS(), "Logistic", "maximize", Xuniform, w0_uniform, w0_uniform),
        (LBFGS(), "Logistic", "maximize", Xuniform, w0_shifted, w0_uniform),
        (LBFGS(), "Logistic", "maximize", Xuniform, w0_very_shifted, w0_uniform),
        (LBFGS(), "Logistic", "maximize", Xshifted, w0_uniform, w0_shifted),
        (LBFGS(), "Logistic", "maximize", Xshifted, w0_shifted, w0_shifted),
        (LBFGS(), "Logistic", "maximize", Xshifted, w0_very_shifted, w0_shifted),
    ],
)
def test_simplex_models(
    optimizer,
    model: Literal["Simplex", "Logistic"],
    direction: Literal["minimize", "maximize"],
    X,
    start,
    expected,
):
    np.seterr(all="raise")

    toymodel = ToyModel(X, direction=direction, model=model)

    snapshot = Snapshot(toymodel, start)
    out_snapshot = optimizer.run(
        snapshot, lambda x, y: print("ITER ==============================")
    )

    assert np.allclose(toymodel.probabilities(out_snapshot.param), expected)