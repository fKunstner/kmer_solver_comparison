from typing import Literal, Union

from kmerexpr.multinomial_model import multinomial_model
from kmerexpr.multinomial_simplex_model import multinomial_simplex_model
from scipy.special import softmax


class SoftmaxModel(multinomial_model):
    def probabilities(self, theta):
        return softmax(theta)

    def logp_grad(self, theta=None, nograd=False):
        f, g = super().logp_grad(theta)
        if nograd:
            return f
        else:
            return f, g


class SimplexModel(multinomial_simplex_model):
    def probabilities(self, theta):
        return theta


KmerModel = Union[SoftmaxModel, SimplexModel]
Simplex = "Simplex"
Softmax = "Softmax"


def get_model(name: str):
    if name not in [Simplex, Softmax]:
        raise ValueError(
            f"Unknown model type {name}, " f"expected one of [{Simplex}, {Softmax}]."
        )
    if name == Simplex:
        return SimplexModel
    else:
        return SoftmaxModel
