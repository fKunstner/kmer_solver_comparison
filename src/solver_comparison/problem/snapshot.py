from dataclasses import dataclass, field
from typing import Optional, Tuple

from numpy.typing import NDArray

from solver_comparison.problem.model import Model


@dataclass
class Snapshot:
    """Snapshot of a model at some parameter.

    Primarily used to cache function/gradient evaluations for logging.
    """

    model: Model
    param: NDArray
    _g: Optional[NDArray] = field(default=None, init=False)
    _f: Optional[float] = field(default=None, init=False)

    def _compute_f_g(self):
        self._f, self._g = self.model.logp_grad(self.param, nograd=False)

    def func(self) -> float:
        if self._f is None:
            self._compute_f_g()
        assert self._f is not None
        return self._f

    def grad(self) -> NDArray:
        if self._g is None:
            self._compute_f_g()
        assert self._g is not None
        return self._g

    def pfg(self) -> Tuple[NDArray, float, NDArray]:
        return self.param, self.func(), self.grad()
