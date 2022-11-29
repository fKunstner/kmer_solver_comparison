import os
from pathlib import Path
from typing import Tuple

from solver_comparison import config


def exp_filepaths(exp_id: str) -> Tuple[str, str, str]:
    exp_folder = os.path.join(config.experiment_dir(), exp_id)
    Path(exp_folder).mkdir(parents=True, exist_ok=True)
    conf_file = os.path.join(exp_folder, "config.json")
    data_file = os.path.join(exp_folder, "data.csv")
    summary_file = os.path.join(exp_folder, "summary.json")
    return conf_file, data_file, summary_file
