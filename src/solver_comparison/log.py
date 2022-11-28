import copy
import logging
import os
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from solver_comparison import config


def normalize_flatten_dict(d: Dict[str, Any], sep: str = "."):
    """Converts a nested dict with string keys to a flat dict."""

    def _normalize_flatten_dict(_d: Dict[str, Any], prefix: Optional[str] = None):
        if not isinstance(_d, dict):
            raise ValueError("Only works on dictionaries")
        for k in _d.keys():
            if not isinstance(k, str):
                raise ValueError(
                    f"Cannot normalize dictionary with non-string key: {k}."
                )
        new_d = {}
        for k, v in _d.items():
            new_k = (prefix + sep + k) if prefix is not None else k
            if isinstance(v, dict):
                new_d.update(_normalize_flatten_dict(v, prefix=new_k))
            else:
                new_d[new_k] = deepcopy(v)
        return new_d

    return _normalize_flatten_dict(d)


def exp_filepaths(exp_id: str) -> Tuple[str, str, str]:
    exp_folder = os.path.join(config.experiment_dir(), exp_id)
    Path(exp_folder).mkdir(parents=True, exist_ok=True)
    conf_file = os.path.join(exp_folder, exp_id + "_config.json")
    data_file = os.path.join(exp_folder, exp_id + "_data.csv")
    summary_file = os.path.join(exp_folder, exp_id + "_summary.json")
    return conf_file, data_file, summary_file


class DataLogger:
    """Tool to log data from an experiment and save the results to disk.

    Mimics the wandb log utility (https://docs.wandb.ai/guides/track/log).

    After initialization, ``log`` can be called with any dictionary to log data.
    Repeated calls to ``log`` will
    - log more data if the keys are different
    - overwrite previous data if the same key is given
    To stop logging for the current step of the experiment,
    call ``end_step`` to commit the results and move on.

    Call ``save`` to save the results to disk.
    The data will be saved as a ``csv`` with the given ``name``
    in the ``datalog_dir`` specified by ``config`` module.
    """

    def __init__(self, exp_id: str, exp_conf: Optional[Dict[str, Any]] = None):
        logging.basicConfig(
            level=config.get_console_logging_level(),
            format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

        self.exp_id: str = exp_id
        self._step: int = 0
        self._exp_conf = {} if exp_conf is None else exp_conf
        self._current_dict: Dict[str, Any] = {}
        self._dicts: List[Dict[str, Any]] = []
        self._summary: Dict[str, Any] = {}

    def end_step(self) -> None:
        """Commits the results for the current step."""
        logging.getLogger(__name__).debug((self._step, self._current_dict))
        self._dicts.append(deepcopy(self._current_dict))
        self._step += 1
        self._current_dict = {}

    def log(self, kwargs: dict) -> None:
        """Log data from the current dictionary.

        Repeated calls to ``log`` without calling ``end_step`` will
        - Overwrite data if the same key is passed
        - Log more data if the keys are new
        Call ``end_step`` to stop logging the current step and move on.

        Args:
            kwargs: dictionary of data to log
        """
        for k, v in kwargs.items():
            self._current_dict[deepcopy(k)] = deepcopy(v)

    def summary(self, kwargs: dict) -> None:
        """Pass a summary dictionary to be saved separately.

        Repeated calls overwrite the values of the summary.

        Args:
            kwargs: dictionary of data to log
        """
        for k, v in kwargs.items():
            self._summary[deepcopy(k)] = deepcopy(v)

    def save(self):
        """Saved the data as a csv.

        The data will be saved in ``name.csv`` in the ``datalog_dir``
        specified by ``config`` module.

        Raises a warning is the DataLogger is saved before changes are
        committed with ``end_step``.
        """

        if len(self._current_dict) > 0:
            warnings.warn(
                "Called save on a DataLogger, "
                "but some data has not been committed using end_step. "
                "The current step will not be saved."
            )

        conf_file, data_file, summary_file = exp_filepaths(self.exp_id)

        logger = logging.getLogger(__name__)
        logger.info(f"Saving config file in {data_file}")
        config_df = pd.DataFrame.from_records([self._dicts])
        config_df.to_csv(conf_file)

        logger.info(f"Saving experiment results in {data_file}")
        data_df = pd.DataFrame.from_records(self._dicts)
        data_df.index.name = "step"
        data_df.to_csv(data_file)

        logger.info(f"Saving summary in {summary_file}")
        summary_df = pd.DataFrame.from_records([self._summary])
        summary_df.to_csv(summary_file)


class RateLimitedLogger:
    """A logger that rate-limits to one message every X seconds or Y call."""

    def __init__(self, time_interval: int = 5, call_interval: Optional[int] = None):
        """Filter calls to ``log`` based on the time and the number of calls.

        The logger only allows one message to be logged every ``time_interval``
        (measured in seconds). If ``call_interval`` is given,

        Args:
            time_interval: Limit messages to one every time_interval
            call_interval: Limit messages to one every call_interval
        """
        self.last_log = None
        self.ignored_calls = 0
        self.time_interval = time_interval
        self.call_interval = call_interval

    def _log(self, *args, **kwargs):
        logging.getLogger(__name__).info(*args, **kwargs)

    def _should_log(self) -> bool:
        def _never_logged():
            return self.last_log is None

        def _should_log_time():
            return time.perf_counter() - self.last_log > self.time_interval

        def _should_log_count():
            return (
                self.call_interval is not None
                and self.ignored_calls >= self.call_interval
            )

        should_log = [_never_logged, _should_log_count, _should_log_time]
        return any(cond() for cond in should_log)

    def log(self, *args, **kwargs):
        """Might pass the arguments to ``getlogger(__name__).log``."""
        if self._should_log():
            self._log(*args, **kwargs)
            self.last_log = time.perf_counter()
            self.ignored_calls = 0
        else:
            self.ignored_calls += 1


class runtime:
    """Timing context manager."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        self.end = time.perf_counter()
        self.time = self.end - self.start


class OnlineSequenceSummary:
    """Returns N equally spaced samples from a sequence of length N.

    Use-case: We are running a process for an unknown number of steps,
    and would like to save the state a few times

    If we knew the process would take T=100 steps and want N=6 save points,
    we would save the value at T=0, 20, 40, 60, 80, 100.
    This utility approximates this when T is unknown.

    Args:
        n (int): Number of samples to save.
            The list will also separately save the first and last value seen.
    """

    def __init__(self, n: int):
        self.first = None
        self.last = None
        self.saved = [None] * n
        self.it_saved = [None] * n
        self.it_to_save = [i + 1 for i in range(n)]
        self.iteration = 0

    def update(self, x):
        if self.iteration == 0:
            self.first = deepcopy(x)

        else:
            if self.iteration > self.it_to_save[-1]:
                self.it_to_save = [2 * i for i in self.it_to_save]

            if self.iteration in self.it_to_save:
                if None in self.it_saved:
                    index = self.it_saved.index(None)
                else:
                    iterations_no_longer_needed = [
                        i
                        for i in self.it_saved
                        if (i is not None and i not in self.it_to_save)
                    ]
                    iteration_to_discard = min(iterations_no_longer_needed)
                    index = self.it_saved.index(iteration_to_discard)

                self.saved[index] = copy.deepcopy(x)
                self.last = self.saved[index]
                self.it_saved[index] = self.iteration
            else:
                self.last = copy.deepcopy(x)
        self.iteration += 1

    def get(self):
        iterations = [0] + [x for x in self.it_saved if x is not None]
        data = [self.first] + [x for x in self.saved if x is not None]

        last_element_already_in_list = (self.iteration - 1) in self.it_saved
        if not last_element_already_in_list and self.last is not None:
            iterations += [self.iteration - 1]
            data += [self.last]

        sort_index = np.argsort(iterations)

        iterations = [iterations[i] for i in sort_index]
        data = [data[i] for i in sort_index]

        return iterations, data


def seconds_to_human_readable(seconds):
    TIME_DURATION_UNITS = (
        ("y", 31536000),
        ("m", 2419200),
        ("w", 604800),
        ("d", 86400),
        ("h", 3600),
        ("m", 60),
        ("s", 1),
    )

    parts = []
    for unit, div in TIME_DURATION_UNITS:
        amount, seconds = divmod(int(seconds), div)
        if amount > 0:
            parts.append(f"{amount}{unit}")
            break
    return " ".join(parts) if len(parts) > 0 else "0s"
