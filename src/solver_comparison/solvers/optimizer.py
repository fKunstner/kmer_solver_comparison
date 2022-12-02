from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from numpy.typing import NDArray

from solver_comparison.problem.model import Model
from solver_comparison.serialization import Serializable

CallbackFunction = Callable[[NDArray, Optional[Dict[str, Any]]], None]


@dataclass
class Optimizer(ABC, Serializable):
    """Base class for optimizers."""

    max_iter: int = 100
    iter: int = field(init=False)

    def __post_init__(self):
        self.iter = 0

    @abstractmethod
    def run(
        self,
        model: Model,
        param: NDArray,
        progress_callback: Optional[CallbackFunction] = None,
    ) -> NDArray:
        raise NotImplementedError
