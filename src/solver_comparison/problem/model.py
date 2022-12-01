from typing import Dict, Literal, Type, Union, get_args

from kmerexpr.multinomial_model import multinomial_model
from kmerexpr.multinomial_simplex_model import multinomial_simplex_model
from scipy.special import softmax


class Logistic(multinomial_model):
    def probabilities(self, theta):
        return softmax(theta)

    def logp_grad(self, theta=None, nograd=False):
        f, g = super().logp_grad(theta)
        if nograd:
            return f
        else:
            return f, g


class Simplex(multinomial_simplex_model):
    def probabilities(self, theta):
        return theta


KmerModel = Union[Logistic, Simplex]
KmerModels = get_args(KmerModel)
KmerModelName = Literal["Logistic", "Simplex"]
model_classes: Dict[KmerModelName, Type[KmerModel]] = {
    "Logistic": Logistic,
    "Simplex": Simplex,
}
