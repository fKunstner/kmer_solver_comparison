import base64
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from solver_comparison.logging import utils
from solver_comparison.logging.datalogger import DataLogger
from solver_comparison.logging.expfiles import exp_filepaths
from solver_comparison.logging.progress_logger import ExperimentProgressLogger
from solver_comparison.logging.sequence_summarizer import OnlineSequenceSummary
from solver_comparison.logging.utils import as_dict, runtime
from solver_comparison.problem.model import Model
from solver_comparison.problem.problem import Problem
from solver_comparison.solvers.initializer import Initializer
from solver_comparison.solvers.optimizer import Optimizer


@dataclass
class Experiment:
    prob: Problem
    opt: Optimizer
    init: Initializer

    def as_dict(self):
        return as_dict(self)

    def hash(self):
        """A hash of the on ``uname`` encoded in b32 (filesystem-safe)."""
        as_bytes = self.uname().encode("ascii")
        as_hash = hashlib.sha256(as_bytes)
        as_b32 = base64.b32encode(as_hash.digest()).decode("ascii")
        return as_b32

    def uname(self):
        """A unique name that can be used to check for equivalence."""
        return base64.b32encode(str(self.as_dict()).encode("ascii")).decode("ascii")

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
        progress_logger = ExperimentProgressLogger(max_iter=self.opt.max_iter)
        saved_parameters = OnlineSequenceSummary(n_to_save=50)

        def info_to_save(param):
            curr_time = time.perf_counter()
            func, grad = model.logp_grad(param, nograd=False)
            return {
                "param": param,
                "func": func,
                "grad_l0": np.linalg.norm(grad, ord=0),
                "grad_l1": np.linalg.norm(grad, ord=1),
                "grad_l2": np.linalg.norm(grad, ord=2),
                "grad_linf": np.linalg.norm(grad, ord=np.inf),
                "time": curr_time - start_time,
            }

        curr_iter = 0

        def progress_callback(
            param: NDArray,
            other: Optional[Dict[str, Any]] = None,
        ):
            curr_time = time.perf_counter()

            nonlocal curr_iter
            curr_iter += 1

            progress_logger.tick(curr_iter=curr_iter, model_and_params=(model, param))

            saved_parameters.update(info_to_save(param))
            func_val, grad_val = model.logp_grad(param, nograd=False)

            datalogger.log(
                {
                    "time": curr_time - start_time,
                    "func_val": func_val,
                    "grad_l0": np.linalg.norm(grad_val, ord=0),
                    "grad_l1": np.linalg.norm(grad_val, ord=1),
                    "grad_l2": np.linalg.norm(grad_val, ord=2),
                    "grad_linf": np.linalg.norm(grad_val, ord=np.inf),
                    "param_l0": np.linalg.norm(param, ord=0),
                    "param_l1": np.linalg.norm(param, ord=1),
                    "param_l2": np.linalg.norm(param, ord=2),
                    "param_linf": np.linalg.norm(param, ord=np.inf),
                }
            )
            if other is not None:
                datalogger.log(other)
            datalogger.end_step()

        # Log initialization
        progress_callback(param)

        param_end = self.opt.run(model, param, progress_callback)

        func, grad = model.logp_grad(param_end, nograd=False)
        saved_iters, saved_values = saved_parameters.get()

        datalogger.summary(
            {
                "prob_end": model.probabilities(param_end).tolist(),
                "obj_end": func,
                "grad_end": grad.tolist(),
                "iter_end": curr_iter,
                "probs": [
                    model.probabilities(x["param"]).tolist() for x in saved_values
                ],
                "params": [x["param"].tolist() for x in saved_values],
                "objs": [x["func"] for x in saved_values],
                "grads_l0": [x["grad_l0"] for x in saved_values],
                "grads_l1": [x["grad_l1"] for x in saved_values],
                "grads_l2": [x["grad_l2"] for x in saved_values],
                "grads_linf": [x["grad_linf"] for x in saved_values],
                "times": [x["time"] for x in saved_values],
                "iters": saved_iters,
            }
        )
        datalogger.save()
