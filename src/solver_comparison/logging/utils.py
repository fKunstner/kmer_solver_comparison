import time
from copy import deepcopy
from typing import Any, Dict, Optional


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


class runtime:
    """Timing context manager."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        self.end = time.perf_counter()
        self.time = self.end - self.start


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
