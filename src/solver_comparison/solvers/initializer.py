from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from solver_comparison.problem.model import Model


@dataclass
class Initializer:
    @abstractmethod
    def initialize_model(self, model: Model) -> NDArray:
        pass


class InitUniform(Initializer):
    def initialize_model(self, model: Model) -> NDArray:
        return np.ones(model.dimension) / model.dimension


class InitZero(Initializer):
    def initialize_model(self, model: Model) -> NDArray:
        return np.zeros(model.dimension)
