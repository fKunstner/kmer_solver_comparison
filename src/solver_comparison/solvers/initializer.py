from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from solver_comparison.problem.model import KmerModel
from solver_comparison.serialization import Serializable


@dataclass
class Initializer(Serializable):
    _METHOD_TYPE = Literal["simplex_uniform", "zero"]
    method: _METHOD_TYPE = "simplex_uniform"

    def initialize_model(self, model: KmerModel) -> NDArray:
        if self.method == "simplex_uniform":
            return np.ones(model.T) / model.T
        elif self.method == "zero":
            return np.zeros(model.T)
        else:
            raise ValueError(
                f"Initialization method {self.method} unknown. "
                f"Expected one of {self._METHOD_TYPE}"
            )
