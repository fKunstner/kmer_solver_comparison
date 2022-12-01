import base64
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from solver_comparison.logging.datalogger import DataLogger
from solver_comparison.logging.expfiles import exp_filepaths
from solver_comparison.logging.progress_logger import ExperimentProgressLogger
from solver_comparison.logging.sequence_summarizer import OnlineSequenceSummary
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

    def has_already_run(self):
        conf_file, data_file, summary_file = exp_filepaths(self.hash())
        return (
            os.path.isfile(conf_file)
            and os.path.isfile(data_file)
            and os.path.isfile(summary_file)
        )

    def _startup(self) -> Tuple[Snapshot, DataLogger]:
        """Loads the model, initialize parameters, create the datalogger."""
        datalogger = DataLogger(exp_id=self.hash(), exp_conf=self.as_dict())

        logger = logging.getLogger(__name__)
        logger.info("Initializing problem")

        with runtime() as loading_time:
            model = self.prob.load_model()

        datalogger.log({"model_load_time": loading_time.time})
        logger.info(f"Problem initialized in {loading_time.time:.2f}s")

        p = self.init.initialize_model(model)
        return Snapshot(model=model, param=p), datalogger

    def run(self):
        curr_p, datalogger = self._startup()

        model = curr_p.model

        start_time = time.perf_counter()
        progress_logger = ExperimentProgressLogger()
        saved_parameters = OnlineSequenceSummary(n=20)

        curr_iter = 0
        max_iter = self.opt.max_iter

        def progress_callback(
            x: Union[Snapshot, NDArray],
            other: Optional[Dict[str, Any]] = None,
        ):
            curr_time = time.perf_counter()

            nonlocal curr_iter
            curr_iter += 1

            snapshot = x if isinstance(x, Snapshot) else Snapshot(model, x)

            progress_logger.tick(
                max_iter=max_iter, curr_iter=curr_iter, snapshot=snapshot
            )
            saved_parameters.update(snapshot.p())

            p, f, g = snapshot.pfg()
            datalogger.log(
                {
                    "time": curr_time - start_time,
                    "f": f,
                    "|g|_1": np.linalg.norm(g, ord=1),
                    "|g|_2": np.linalg.norm(g, ord=2),
                    "|g|_inf": np.linalg.norm(g, ord=np.inf),
                    "|p|_1": np.linalg.norm(p, ord=1),
                    "|p|_2": np.linalg.norm(p, ord=2),
                    "|p|_inf": np.linalg.norm(p, ord=np.inf),
                }
            )
            if other is not None:
                datalogger.log(other)
            datalogger.end_step()

        curr_p = self.opt.run(curr_p, progress_callback)

        datalogger.summary(
            {
                "x": curr_p.model.probabilities(curr_p.param).tolist(),
                "loss_records": curr_p.f(),
                "iteration_counts": curr_iter,
                "grad": curr_p.g().tolist(),
                "xs": saved_parameters.get(),
            }
        )
        datalogger.save()
