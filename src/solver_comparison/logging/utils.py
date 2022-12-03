import re
import time
import unicodedata
from copy import deepcopy
from typing import Any, Dict, Optional


def normalize_flatten_dict(a_dict: Dict[str, Any], sep: str = "."):
    """Converts a nested dict with string keys to a flat dict."""

    def _normalize_flatten_dict(_a_dict: Dict[str, Any], prefix: Optional[str] = None):
        if not isinstance(_a_dict, dict):
            raise ValueError("Only works on dictionaries")
        for key in _a_dict.keys():
            if not isinstance(key, str):
                raise ValueError(
                    f"Cannot normalize dictionary with non-string key: {key}."
                )
        new_d = {}
        for key, val in _a_dict.items():
            new_k = (prefix + sep + key) if prefix is not None else key
            if isinstance(val, dict):
                new_d.update(_normalize_flatten_dict(val, prefix=new_k))
            else:
                new_d[new_k] = deepcopy(val)
        return new_d

    return _normalize_flatten_dict(a_dict)


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


def slugify(s: str):
    """Convert to a filesystem-safe ASCII string."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.replace("[", "(").replace("]", ")")
    s = re.sub(r"[^\w\s()-]", "", s.lower())
    return re.sub(r"[-\s]+", "-", s).strip("-_")
