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

        param = self.init.initialize_model(model)
        return Snapshot(model=model, param=param), datalogger

    def run(self):
        curr_p, datalogger = self._startup()

        model = curr_p.model

        start_time = time.perf_counter()
        progress_logger = ExperimentProgressLogger()
        saved_parameters = OnlineSequenceSummary(n_to_save=20)

        curr_iter = 0
        max_iter = self.opt.max_iter

        def progress_callback(
            param_or_snap: Union[Snapshot, NDArray],
            other: Optional[Dict[str, Any]] = None,
        ):
            curr_time = time.perf_counter()

            nonlocal curr_iter
            curr_iter += 1

            snapshot = (
                param_or_snap
                if isinstance(param_or_snap, Snapshot)
                else Snapshot(model, param_or_snap)
            )

            progress_logger.tick(
                max_iter=max_iter, curr_iter=curr_iter, snapshot=snapshot
            )
            saved_parameters.update(snapshot.param)

            param, func_val, grad_val = snapshot.pfg()
            datalogger.log(
                {
                    "time": curr_time - start_time,
                    "func_val": func_val,
                    "|grad_val|_1": np.linalg.norm(grad_val, ord=1),
                    "|grad_val|_2": np.linalg.norm(grad_val, ord=2),
                    "|grad_val|_inf": np.linalg.norm(grad_val, ord=np.inf),
                    "|param|_1": np.linalg.norm(param, ord=1),
                    "|param|_2": np.linalg.norm(param, ord=2),
                    "|param|_inf": np.linalg.norm(param, ord=np.inf),
                }
            )
            if other is not None:
                datalogger.log(other)
            datalogger.end_step()

        # Log initialization
        progress_callback(curr_p)

        end_snapshot = self.opt.run(curr_p, progress_callback)

        datalogger.summary(
            {
                "x": end_snapshot.model.probabilities(end_snapshot.param).tolist(),
                "loss_records": end_snapshot.func(),
                "iteration_counts": curr_iter,
                "grad": end_snapshot.grad().tolist(),
                "xs": saved_parameters.get(),
            }
        )
        datalogger.save()
