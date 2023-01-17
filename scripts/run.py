import argparse
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from solver_comparison.plotting import make_all_plots

experiments = []


def load_module(file_path: Path):
    """Import a module by filepath."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Experiment file {file_path}.")

    spec = spec_from_file_location(f"UserPlottingCode.{file_path.stem}", file_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import experiments in '{file_path}'")

    module = module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ImportError(f"Could not import experiments in '{file_path}'") from exc

    return module


def load_experiments(file_path: Path):
    module = load_module(file_path)
    if not hasattr(module, "experiments"):
        raise ValueError(
            f"File {file_path} does not define a variable called 'experiments'."
        )
    return module.experiments


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Run an experiment configuration file",
    )
    parser.add_argument(
        "exp_config",
        nargs="?",
        type=Path,
        default=None,
        help="The configuration file containing a variable `experiments`",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli()
    experiments = load_experiments(args.exp_config)

    for exp in experiments:
        print(exp.as_dict())
        if not exp.has_already_run():
            exp.run()
        print("stored at ", exp.hash())

    make_all_plots(experiments)
