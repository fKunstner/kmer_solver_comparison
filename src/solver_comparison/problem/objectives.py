import numpy as np
from kmerexpr.multinomial_model import multinomial_model
from kmerexpr.multinomial_simplex_model import multinomial_simplex_model
from numpy.typing import NDArray
from scipy.special import softmax


class Objective:
    pass


class MultinomialSimplexObjective(Objective):
    def __init__(self, kmer_model: multinomial_simplex_model):
        self._model = kmer_model
        if self._model.beta != 1.0:
            raise ValueError("Priors (using beta != 1.0) not yet implemented")

    def _func(self, psi_probabilities: NDArray):
        return self._model.ynnz.dot(np.log(psi_probabilities))

    def _grad(self, psi_probabilities: NDArray):
        weights = self._model.ynnz / psi_probabilities
        return weights @ self._model.xnnz

    def _psi_probabilities(self, param) -> NDArray:
        return self._model.xnnz.dot(param)

    def func(self, param: NDArray):
        return self._func(self._psi_probabilities(param))

    def grad(self, param: NDArray):
        return self._func(self._psi_probabilities(param))

    def func_and_grad(self, param: NDArray):
        psi_probs = self._psi_probabilities(param)
        return self._func(psi_probs), self._grad(psi_probs)

    def objfunc_along_direction(self, param, direction):
        probs = self._psi_probabilities(param)
        dprobs = self._model.xnnz.dot(direction)

        def obj_for_stepsize(ss: float):
            return self._func(probs + ss * dprobs)

        return obj_for_stepsize


class MultinomialObjective(Objective):
    def __init__(self, kmer_model: multinomial_model):
        self._model = kmer_model

    def _func(self, param: NDArray, psi_probabilities: NDArray):
        log_likelihood = self._model.ynnz.dot(np.log(psi_probabilities))
        log_prior = -self._model.beta * param.dot(param)
        return log_likelihood + log_prior

    def _grad(self, param: NDArray, probs: NDArray, psi_probabilities: NDArray):
        weights = self._model.ynnz / psi_probabilities
        g_log_likelihood = self._model.xnnz.T @ weights * probs - self._model.N * probs
        g_prior = -2 * self._model.beta * param
        return g_log_likelihood + g_prior

    def _psi_probabilities(self, probs: NDArray) -> NDArray:
        return self._model.xnnz.dot(probs)

    def func(self, param: NDArray):
        probs = softmax(param)
        return self._func(param, self._psi_probabilities(probs))

    def grad(self, param: NDArray):
        probs = softmax(param)
        return self._grad(param, probs, self._psi_probabilities(probs))

    def func_and_grad(self, param: NDArray):
        probs = softmax(param)
        psi_probs = self._psi_probabilities(probs)
        return self._func(param, psi_probs), self._grad(param, probs, psi_probs)

    def objfunc_along_direction(self, param, direction):
        raise NotImplementedError
