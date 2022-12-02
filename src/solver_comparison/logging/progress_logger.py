import time
from typing import Optional

import numpy as np

from solver_comparison.logging.rate_limited_logger import RateLimitedLogger
from solver_comparison.logging.utils import seconds_to_human_readable
from solver_comparison.problem.snapshot import Snapshot


class ExperimentProgressLogger:
    def __init__(self, log_every: int = 3):
        self.timelogger = RateLimitedLogger(time_interval=log_every)
        self.start_time = time.perf_counter()

    def tick(self, max_iter: int, curr_iter: int, snapshot: Optional[Snapshot]):
        i = curr_iter
        progress = curr_iter / max_iter
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
        if snapshot is not None:
            func, grad = snapshot.func(), snapshot.grad()
            data_str = f"Loss={func:.2e}, gnorm={np.linalg.norm(grad):.2e}"

        self.timelogger.log(f"{iter_str } [{time_str:>18}] - {data_str}")
