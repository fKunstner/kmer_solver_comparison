import base64
import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict

from solver_comparison.logging.datalogger import DataLogger
from solver_comparison.logging.expfiles import exp_filepaths
from solver_comparison.logging.progress_logger import ExperimentProgressLogger
from solver_comparison.logging.utils import runtime
from solver_comparison.problem.problem import Problem
from solver_comparison.problem.snapshot import Snapshot
from solver_comparison.serialization import Serializable
from solver_comparison.solvers.initializer import Initializer
from solver_comparison.solvers.optimizer import Optimizer


@dataclass
class Experiment(Serializable):
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

    def run(self):
        curr_p = self._startup()

        progress_logger = ExperimentProgressLogger()

        def progress_callback(
            snapshot: Snapshot, curr_iter: int, max_iter: int, other: Dict[str, Any]
        ):
            progress_logger.tick(
                max_iter=max_iter, current_iter=curr_iter, snapshot=snapshot
            )

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
