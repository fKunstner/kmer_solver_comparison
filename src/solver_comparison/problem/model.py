from typing import Literal, Union

from kmerexpr.multinomial_model import multinomial_model
from kmerexpr.multinomial_simplex_model import multinomial_simplex_model
from scipy.special import softmax


class SoftmaxModel(multinomial_model):
    def probabilities(self, theta):
        return softmax(theta)

    def logp_grad(self, theta=None, nograd=False):
        func, grad = super().logp_grad(theta)
        if nograd:
            return func
        else:
            return func, grad


class SimplexModel(multinomial_simplex_model):
    def probabilities(self, theta):
        return theta


KmerModel = Union[SoftmaxModel, SimplexModel]
SIMPLEX = "Simplex"
SOFTMAX = "Softmax"


def get_model(name: str):
    if name not in [SIMPLEX, SOFTMAX]:
        raise ValueError(
            f"Unknown model type {name}, " f"expected one of [{SIMPLEX}, {SOFTMAX}]."
        )
    if name == SIMPLEX:
        return SimplexModel
    else:
        return SoftmaxModel
