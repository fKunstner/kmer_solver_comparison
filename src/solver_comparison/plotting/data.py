import json

from solver_comparison.experiment import Experiment
from solver_comparison.logging.expfiles import exp_filepaths


def convert_summary_to_dict_results(summary):
    dict_results = {
        "x": summary["prob_end"],
        "xs": summary["probs"],
        # Todo: "Loss" is inaccurate, it's an objective and higher is better
        # Requires a fix in kmerexpr
        "loss_records": summary["objs"],
        "iteration_counts": summary["iters"],
        "grads_l0": summary["grads_l0"],
        "grads_l1": summary["grads_l1"],
        "grads_l2": summary["grads_l2"],
        "grads_linf": summary["grads_linf"],
    }
    return dict_results


def load_dict_result(exp: Experiment):
    conf_path, data_path, summary_path = exp_filepaths(exp.hash())
    with open(summary_path, "r") as fp:
        summary = json.load(fp)
    return convert_summary_to_dict_results(summary)


def get_shortname(exp):
    return exp.opt.__class__.__name__


def get_plot_base_filename(exp: Experiment, with_optimizer: bool = True):
    """Generate base filename for the experiment.

    Version of `kmerexpr.plotting.get_plot_title` for all optimizers.
    """
    problem, optimizer, initializer = exp.prob, exp.opt, exp.init

    title = (
        f"{problem.filename}-{problem.model_type}-"
        f"N-{problem.N}-L-{problem.L}-K-{problem.K}-a-{problem.alpha}"
    )

    if with_optimizer:
        title += f"-init-{initializer.__class__.__name__}-{exp.opt.__class__.__name__}"

    return title
