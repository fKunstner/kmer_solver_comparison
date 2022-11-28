import os


def _key(name: str) -> str:
    return f"KMEREXPR_BENCH_{name}"


K_LOGLEVEL = _key("LOGLEVEL")
K_DATA_ROOT = _key("DATA_ROOT")

DEFAULTS = {
    K_DATA_ROOT: os.path.expanduser(os.path.join("~", "kmer_solver_comparison")),
    K_LOGLEVEL: "INFO",
}


def _get_env_var(key: str) -> str:
    val = os.environ.get(key, None)
    if val is None:
        raise EnvironmentError(
            f"Environment variable {key} undefined. "
            f"See readme for how to set environment variable."
        )
    return val


def data_root() -> str:
    return os.environ.get(K_DATA_ROOT, DEFAULTS[K_DATA_ROOT])


def workspace() -> str:
    return os.path.join(os.path.realpath(data_root()), "workspace")


def dataset_dir() -> str:
    return os.path.join(os.path.realpath(data_root()), "data")


def log_dir() -> str:
    return os.path.join(workspace(), "logs")


def datalog_dir() -> str:
    return os.path.join(workspace(), "data_logs")


def get_console_logging_level():
    return _get_env_var(K_LOGLEVEL)


def experiment_dir():
    return os.path.join(workspace(), "results")
