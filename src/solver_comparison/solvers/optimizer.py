from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, Optional, Union

import numpy as np
from numpy.typing import NDArray
from solver_comparison.problem.snapshot import Snapshot
from solver_comparison.serialization import Serializable

CallbackFunction = Callable[[Union[Snapshot, NDArray], Optional[Dict[str, Any]]], None]


@dataclass
class Optimizer(ABC, Serializable):
    """Base class for optimizers."""

    max_iter: int = 100
    p_tol: float = 10 ** -16
    g_tol: float = 10 ** -16
    f_tol: float = 10 ** -16
    iter: int = field(init=False)
    solver_name: ClassVar[str] = "generic_optimizer"

    def __post_init__(self):
        self.iter = 0

    def should_stop(self, curr_p: Snapshot, old_p: Snapshot):
        df, dp, dg = (
            curr_p.f() - old_p.f(),
            curr_p.p() - old_p.p(),
            curr_p.g() - old_p.g(),
        )
        i_check = self.iter > self.max_iter
        f_check = np.abs(df) <= self.f_tol
        p_check = np.linalg.norm(dp) ** 2 <= self.p_tol
        g_check = np.linalg.norm(dg) ** 2 <= self.g_tol
        return any([i_check, f_check, p_check, g_check])

    @abstractmethod
    def run(
        self,
        curr_p: Snapshot,
        progress_callback: CallbackFunction,
    ) -> Snapshot:
        raise NotImplementedError
