import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from solver_comparison.logging.sequence_summarizer import OnlineSequenceSummary
from solver_comparison.problem.snapshot import Snapshot
from solver_comparison.solvers.optimizer import CallbackFunction, Optimizer


@dataclass
class GDLS(Optimizer):
    """Gradient Descent using a Backtracking Armijo Linesearch.

    Looks for sufficient progress under Euclidean smoothness.

    Args:
        c: The strength of the Armijo condition, (0, 1).
           Larger values ask for more progress but is more difficult to satisfy.
        max: Maximum step-size allowed and starting point for backtracking.
        decr: Multiplicative factor when backtracking, (0, 1)
        incr: Multiplicative factor at the start of a new iteration [1, ...)
        max_iter: Maximum number of inner iterations
    """

    c: float = 0.5
    max: float = 10**10
    decr: float = 0.5
    incr: float = 1.0
    max_iter: int = 100
    curr_ss: float = field(init=False)
    solver_name: ClassVar[str] = "gd_linesearch"

    def __post_init__(self):
        super().__post_init__()
        if not (0 < self.c < 1):
            raise ValueError(f"Strength parameter should be 0 < c < 1. Got {self.c}")
        if not (0 < self.decr < 1):
            raise ValueError(f"Mult. decr should be 0 < decr < 1. Got {self.decr}")
        if not (1.0 <= self.incr):
            raise ValueError(f"Mult. incr should be 1 <= incr. Got {self.incr}")
        if not isinstance(self.max_iter, int) or not (0 < self.max_iter):
            raise ValueError(f"max_iter should be an integer > 0. Got {self.incr}")
        self.curr_ss = self.max

    def run(
        self,
        curr_p: Snapshot,
        progress_callback: CallbackFunction,
    ) -> Tuple[Snapshot, int, OnlineSequenceSummary]:
        saved_parameters = OnlineSequenceSummary(n=20)
        saved_parameters.update(curr_p)

        t = 0
        for t in range(self.max_iter):
            old_p = curr_p
            curr_p, to_log = self.step(curr_p)

            progress_callback(curr_p, t, self.max_iter, to_log)

            saved_parameters.update(curr_p.param)

            if self.should_stop(curr_p, old_p):
                break

        return curr_p, t, saved_parameters

    def step(self, current: Snapshot) -> Tuple[Snapshot, Dict[str, Any]]:
        def newpoint(ss: float) -> NDArray:
            return current.param - ss * current.g()

        f_curr, g_curr = current.f(), current.g()
        curr_grad_norm = self.c * np.linalg.norm(g_curr) ** 2
        self.curr_ss = self.incr * self.curr_ss

        stepsizes: List[float] = []
        function_values: List[float] = []
        found = False
        new = None
        for t in range(self.max_iter):
            new = Snapshot(param=newpoint(self.curr_ss), model=current.model)

            try:
                f_new = new.f()
            except FloatingPointError:
                logging.getLogger(__name__).debug(
                    f"Overflow in linesearch with step-size {self.curr_ss:.2e}"
                )
                f_new = np.inf

            stepsizes.append(self.curr_ss)
            function_values.append(f_new)

            sufficient_decrease = f_new < f_curr - self.curr_ss * curr_grad_norm
            if sufficient_decrease:
                found = True
                break
            else:
                self.curr_ss = self.curr_ss * self.decr

        to_log = {"stepsizes": stepsizes, "function_values": function_values}

        if not found:
            logging.getLogger(__name__).warning(
                "LineSearch terminated without finding appropriate parameter."
            )
            return current, to_log
        else:
            assert new is not None
            return new, to_log
