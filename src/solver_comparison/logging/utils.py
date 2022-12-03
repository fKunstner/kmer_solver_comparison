import json
import re
import time
import unicodedata
from copy import deepcopy
from dataclasses import fields, is_dataclass
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


def as_dict(x):
    """Gets a dictionary representation of a DataClass.

    A re-implementation of DataClass's ``asdict`` ignoring non-init
    fields.
    """

    def _recursive_as_dict(_obj):
        """Gets a representation of obj.

        A re-implementation of DataClass's ``asdict`` ignoring non-init
        fields.
        """
        if is_dataclass(_obj):
            return (
                _obj.__class__.__name__,
                {
                    f.name: _recursive_as_dict(getattr(_obj, f.name))
                    for f in fields(_obj)
                    if f.init
                },
            )
        elif isinstance(_obj, tuple) and hasattr(_obj, "_fields"):
            return type(_obj)(*[_recursive_as_dict(v) for v in _obj])
        elif isinstance(_obj, (list, tuple)):
            return type(_obj)(_recursive_as_dict(v) for v in _obj)
        elif isinstance(_obj, dict):
            return type(_obj)(
                (_recursive_as_dict(k), _recursive_as_dict(v)) for k, v in _obj.items()
            )
        else:
            return _obj

    return _recursive_as_dict(x)


def str_to_dict(some_str):
    return json.loads(some_str)


def dict_to_str(some_dict):
    return json.dumps(some_dict, sort_keys=True)
