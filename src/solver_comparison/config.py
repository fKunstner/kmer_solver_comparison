import os


def _key(name: str) -> str:
    return f"KMEREXPR_BENCH_{name}"


K_LOGLEVEL = _key("LOGLEVEL")
K_WORKSPACE = _key("WORKSPACE")

DEFAULTS = {
    K_WORKSPACE: os.path.expanduser(os.path.join("~", "kmer_solver_comparison")),
    K_LOGLEVEL: "INFO",
}


def workspace() -> str:
    return os.environ.get(K_WORKSPACE, DEFAULTS[K_WORKSPACE])


def dataset_dir() -> str:
    return os.path.join(os.path.realpath(workspace()), "data")


def experiment_dir():
    return os.path.join(workspace(), "results")


def get_console_logging_level():
    return os.environ.get(K_LOGLEVEL, DEFAULTS[K_LOGLEVEL])
