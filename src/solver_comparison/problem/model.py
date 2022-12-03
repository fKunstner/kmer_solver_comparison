from abc import abstractmethod
from typing import Optional

import numpy as np
from kmerexpr.multinomial_model import multinomial_model
from kmerexpr.multinomial_simplex_model import multinomial_simplex_model
from numpy.typing import NDArray
from scipy.special import softmax

from solver_comparison.problem.objectives import (
    MultinomialObjective,
    MultinomialSimplexObjective,
    Objective,
)

SIMPLEX = "Simplex"
SOFTMAX = "Softmax"


def unknown_model_error(name):
    raise (f"Unknown model type {name}, expected one of [{SIMPLEX}, {SOFTMAX}].")


def get_model(name: str):
    if name not in [SIMPLEX, SOFTMAX]:
        raise ValueError(unknown_model_error(name))
    if name == SIMPLEX:
        return multinomial_simplex_model
    else:
        return multinomial_model


class _FunctionGradientCache:
    def __init__(self):
        self.input = None
        self.func_cache = None
        self.grad_cache = None

    def is_in_cache(self, theta: NDArray, nograd) -> bool:
        if (self.input == theta).all():
            if nograd:
                return self.func_cache is not None
            else:
                return self.func_cache is not None and self.grad_cache is not None
        return False

    def cache(self, param, func, grad):
        self.input = param.copy()
        self.func_cache = func
        self.grad_cache = grad

    def cached_values(self, nograd):
        if nograd:
            return self.func_cache
        else:
            return self.func_cache, self.grad_cache


class Model:
    def __init__(self, model_type: str, x_file, y_file, beta=1.0, lengths=None):
        self.model_type = model_type
        self.kmerexpr_model = get_model(model_type)(
            x_file=x_file,
            y_file=y_file,
            beta=beta,
            lengths=lengths,
            solver_name=None,
        )

        self._objective: Objective
        if isinstance(self.kmerexpr_model, multinomial_simplex_model):
            self._objective = MultinomialSimplexObjective(self.kmerexpr_model)
        elif isinstance(self.kmerexpr_model, multinomial_model):
            self._objective = MultinomialObjective(self.kmerexpr_model)
        else:
            raise unknown_model_error(self.kmerexpr_model.__class__.__name__)

        self._cache = _FunctionGradientCache()

    def logp_grad(self, param, nograd=False, Hessinv=False):
        if self._cache.is_in_cache(param, nograd):
            return self._cache.cached_values(nograd)

        if Hessinv:
            raise NotImplementedError("Hessian preconditioning not implemented")

        if nograd:
            func, grad = self._objective.func(param), None
        else:
            func, grad = self._objective.func_and_grad(param)

        self._cache.cache(param, func, grad)
        return self._cache.cached_values(nograd)

    def objfunc_along_direction(self, param, direction):
        """Returns a function to evaluate stepsizes along given direction.

        For simplex models, assumes the step-size is [0, 1].
        """
        return self._objective.objfunc_along_direction(param, direction)

    def probabilities(self, params):
        if self.model_type == SIMPLEX:
            return params
        elif self.model_type == SOFTMAX:
            return softmax(params)
        else:
            raise ValueError(unknown_model_error(self.model_type))

    @property
    def dimension(self) -> int:
        return int(self.kmerexpr_model.T)
