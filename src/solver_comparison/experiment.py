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
from solver_comparison.problem.model import Model
from solver_comparison.problem.problem import Problem
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

    def _startup(self) -> Tuple[Model, NDArray, DataLogger]:
        """Loads the model, initialize parameters, create the datalogger."""
        datalogger = DataLogger(exp_id=self.hash(), exp_conf=self.as_dict())

        logger = logging.getLogger(__name__)
        logger.info("Initializing problem")

        with runtime() as loading_time:
            model = self.prob.load_model()

        datalogger.log({"model_load_time": loading_time.time})
        logger.info(f"Problem initialized in {loading_time.time:.2f}s")

        param = self.init.initialize_model(model)
        return model, param, datalogger

    def run(self):
        model, param, datalogger = self._startup()

        start_time = time.perf_counter()
        progress_logger = ExperimentProgressLogger()
        saved_parameters = OnlineSequenceSummary(n_to_save=20)

        curr_iter = 0
        max_iter = self.opt.max_iter

        def progress_callback(
            param: NDArray,
            other: Optional[Dict[str, Any]] = None,
        ):
            curr_time = time.perf_counter()

            nonlocal curr_iter
            curr_iter += 1

            progress_logger.tick(
                max_iter=max_iter, curr_iter=curr_iter, model_and_params=(model, param)
            )
            saved_parameters.update(param)

            func_val, grad_val = model.logp_grad(param, nograd=False)

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
        progress_callback(param)

        param_end = self.opt.run(model, param, progress_callback)
        func, grad = model.logp_grad(param_end, nograd=False)
        datalogger.summary(
            {
                "x": model.probabilities(param_end).tolist(),
                "loss_records": func,
                "iteration_counts": curr_iter,
                "grad": grad,
                "xs": saved_parameters.get(),
            }
        )
        datalogger.save()
