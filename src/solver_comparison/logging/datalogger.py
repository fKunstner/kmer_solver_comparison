import logging
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional

import pandas as pd

from solver_comparison import config
from solver_comparison.logging.expfiles import exp_filepaths


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
            format="[%(name)s %(asctime)s] %(levelname)s: %(message)s",
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
        config_df = pd.DataFrame.from_records([self._exp_conf])
        config_df.to_csv(conf_file)

        logger.info(f"Saving experiment results in {data_file}")
        data_df = pd.DataFrame.from_records(self._dicts)
        data_df.index.name = "step"
        data_df.to_csv(data_file)

        logger.info(f"Saving summary in {summary_file}")
        summary_df = pd.DataFrame.from_records([self._summary])
        summary_df.to_csv(summary_file)
