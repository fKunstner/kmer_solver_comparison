import base64
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from solver_comparison.expconf import ExpConf
from solver_comparison.log import (
    DataLogger,
    OnlineSequenceSummary,
    RateLimitedLogger,
    exp_filepaths,
    runtime,
    seconds_to_human_readable,
)
from solver_comparison.problem.problem import Problem
from solver_comparison.problem.snapshot import Snapshot
from solver_comparison.solvers.initializer import Initializer
from solver_comparison.solvers.optimizer import Optimizer


@dataclass
class Experiment(ExpConf):
    prob: Problem
    opt: Optimizer
    init: Initializer

    def hash(self):
        """A hash of the on ``uname`` encoded in b32 (filesystem-safe)."""
        as_bytes = self.uname().encode("ascii")
        as_hash = hashlib.sha256(as_bytes)
        as_b32 = base64.b32encode(as_hash.digest()).decode("ascii")
        return as_b32

    def uname(self):
        """A unique name that can be used to check for equivalence."""
        return base64.b32encode(self.as_str().encode("ascii")).decode("ascii")

    def _startup(self):
        self.datalogger = DataLogger(exp_id=self.hash(), exp_conf=self.as_dict())

        logger = logging.getLogger(__name__)
        logger.info("Initializing problem")

        with runtime() as loading_time:
            model = self.prob.load_model()

        self.datalogger.log({"model_load_time": loading_time.time})
        logger.info(f"Problem initialized in {loading_time.time:.2f}s")
        p = self.init.initialize_model(model)
        return Snapshot(model=model, param=p)

    @staticmethod
    def dummy_callback(*args, **kwargs):
        pass

    def run(
        self,
        progress_callback: Optional[
            Callable[[int, float, Optional[Snapshot]], None]
        ] = None,
    ):
        curr_p = self._startup()

        if progress_callback is None:
            progress_callback = ExperimentMonitor().callback

        curr_p, t, saved_parameters = self.opt.run(
            curr_p, progress_callback, self.datalogger
        )

        self.datalogger.summary(
            {
                "x": curr_p.model.probabilities(curr_p.param).tolist(),
                "loss_records": curr_p.f(),
                "iteration_counts": t,
                "grad": curr_p.g().tolist(),
                "xs": saved_parameters.get(),
            }
        )
        self.datalogger.save()

    def has_already_run(self):
        conf_file, data_file, summary_file = exp_filepaths(self.hash())
        return (
            os.path.isfile(conf_file)
            and os.path.isfile(data_file)
            and os.path.isfile(summary_file)
        )


class ExperimentMonitor:
    def __init__(self, log_every: int = 3):
        self.timelogger = RateLimitedLogger(time_interval=log_every)
        self.start_time = time.perf_counter()

    def callback(self, max_iter: int, progress: float, snap: Optional[Snapshot]):
        i = int(max_iter * progress)
        i_width = len(str(max_iter))
        iter_str = f"Iter {i: >{i_width}}/{max_iter}"

        time_str = ""
        if self.start_time is not None:
            run_s = time.perf_counter() - self.start_time
            run_h = seconds_to_human_readable(run_s)

            eta_h, rem_h = "?", "?"
            if progress > 0:
                eta_s = run_s / progress
                rem_s = eta_s - run_s
                eta_h = seconds_to_human_readable(eta_s)
                rem_h = seconds_to_human_readable(rem_s)

            time_str = f"{run_h:>3}/{eta_h:>3} ({rem_h:>3} rem.)"

        data_str = ""
        if snap is not None:
            f, g = snap.f(), snap.g()
            data_str = f"Loss={f:.2e}  gnorm={np.linalg.norm(g):.2e}"

        self.timelogger.log(f"{iter_str } [{time_str:>18}] - {data_str}")
