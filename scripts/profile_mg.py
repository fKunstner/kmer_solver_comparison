import timeit

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.sparse
from numpy.typing import NDArray

from solver_comparison.problem import problem_settings
from solver_comparison.problem.model import SIMPLEX
from solver_comparison.problem.objectives import MultinomialSimplexObjective
from solver_comparison.problem.problem import Problem


def load_data():
    sx = sp.sparse.load_npz("sparse_x.npz")
    dx = sp.sparse.load_npz("dense_x.npz")
    sy = np.load("sparse_y.npy")
    dy = np.load("dense_y.npy")
    return {
        "sparse": (sx, sy),
        "dense": (dx, dy),
    }


def make_figure(fig, data):
    class Model:
        def __init__(self, data):
            self.xnnz = data[0]
            self.ynnz = data[1]
            self.beta = 1.0

    m1, m2 = Model(data["sparse"]), Model(data["dense"])

    obj1 = MultinomialSimplexObjective(m1)
    obj2 = MultinomialSimplexObjective(m2)
    d = m1.xnnz.shape[1]
    p = np.ones(d) / d
    p = p * obj1._grad(obj1._psi_probabilities(p))
    p2 = np.ones(d) / d
    p2 = p2 * obj2._grad(obj2._psi_probabilities(p2))

    class MultinomialSimplexObjective2:
        def __init__(self, kmer_model):
            self._model = Model([kmer_model.xnnz.tocsr(), kmer_model.ynnz])
            self.qnnz = self._model.ynnz / np.sum(self._model.ynnz)
            if self._model.beta != 1.0:
                raise ValueError("Priors (using beta != 1.0) not yet implemented")

        def _func(self, psi_probabilities: NDArray):
            return self.qnnz.dot(np.log(psi_probabilities))

        def _grad(self, psi_probabilities: NDArray):
            weights = self.qnnz / psi_probabilities
            return weights @ self._model.xnnz

        def _psi_probabilities(self, param) -> NDArray:
            return self._model.xnnz.dot(param)

        def func(self, param: NDArray):
            return self._func(self._psi_probabilities(param))

        def grad(self, param: NDArray):
            return self._grad(self._psi_probabilities(param))

        def func_and_grad(self, param: NDArray):
            psi_probs = self._psi_probabilities(param)
            return self._func(psi_probs), self._grad(psi_probs)

        def objfunc_along_direction(self, param, direction):
            probs = self._psi_probabilities(param)
            dprobs = self._model.xnnz.dot(direction)

            def obj_for_stepsize(ss: float):
                return self._func(probs + ss * dprobs)

            return obj_for_stepsize

    obj12 = MultinomialSimplexObjective2(m1)
    obj22 = MultinomialSimplexObjective2(m2)

    r = 20
    print("For sparse model")
    print("Func Before: ", timeit.timeit(lambda: obj1.func(p), number=r))
    print("Func After:  ", timeit.timeit(lambda: obj12.func(p), number=r))
    print("Grad Before: ", timeit.timeit(lambda: obj1.grad(p), number=r))
    print("Grad After:  ", timeit.timeit(lambda: obj12.grad(p), number=r))
    print("Both Before: ", timeit.timeit(lambda: obj1.func_and_grad(p), number=r))
    print("Both After:  ", timeit.timeit(lambda: obj12.func_and_grad(p), number=r))
    r = 5
    print("For dense model")
    print("Func Before: ", timeit.timeit(lambda: obj2.func(p2), number=r))
    print("Func After:  ", timeit.timeit(lambda: obj22.func(p2), number=r))
    print("Grad Before: ", timeit.timeit(lambda: obj2.grad(p2), number=r))
    print("Grad After:  ", timeit.timeit(lambda: obj22.grad(p2), number=r))
    print("Both Before: ", timeit.timeit(lambda: obj2.func_and_grad(p2), number=r))
    print("Both After:  ", timeit.timeit(lambda: obj22.func_and_grad(p2), number=r))


if __name__ == "__main__":

    generate_data = False

    if generate_data:
        filename, K, N, L, alpha = problem_settings["medium-sparse"]
        sparse_problem = Problem(
            model_type=SIMPLEX, filename=filename, K=K, N=N, L=L, alpha=alpha, beta=1.0
        )
        sparse_model = sparse_problem.load_model()
        sp.sparse.save_npz("sparse_x.npz", sparse_model.kmerexpr_model.xnnz)
        np.save("sparse_y.npy", sparse_model.kmerexpr_model.ynnz)

        filename, K, N, L, alpha = problem_settings["medium-dense"]
        dense_problem = Problem(
            model_type=SIMPLEX, filename=filename, K=K, N=N, L=L, alpha=alpha, beta=1.0
        )
        dense_model = dense_problem.load_model()
        sp.sparse.save_npz("dense_x.npz", dense_model.kmerexpr_model.xnnz)
        np.save("dense_y.npy", dense_model.kmerexpr_model.ynnz)
    else:
        data = load_data()
        fig = plt.figure()
        make_figure(fig, data)
        fig.show()
        fig.close()
