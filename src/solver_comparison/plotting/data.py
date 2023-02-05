import json
from functools import lru_cache
from typing import List

import numpy as np
from kmerexpr.utils import load_lengths
from scipy.stats import entropy

from solver_comparison.experiment import Experiment
from solver_comparison.logging.expfiles import exp_filepaths
from solver_comparison.problem.problem import Problem


def convert_summary_to_dict_results(summary):
    dict_results = {
        "x": summary["prob_end"],
        "xs": summary["probs"],
        "times": summary["times"],
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


def get_plot_dirname(exp: Experiment):
    return (
        f"{exp.prob.filename}"
        f"_N{exp.prob.N}"
        f"_L{exp.prob.L}"
        f"_K{exp.prob.K}"
        f"_a{exp.prob.alpha}"
    )


def get_opt_full_name(exp: Experiment):
    problem, optimizer, initializer = exp.prob, exp.opt, exp.init
    return (
        f"{exp.opt.__class__.__name__}["
        f"model{problem.model_type}"
        + f"_init{initializer.__class__.__name__}"
        + f"_maxiter{exp.opt.max_iter}]"
    )


def get_plot_base_filename(exp: Experiment, with_optimizer: bool = True):
    """Generate base filename for the experiment.

    Version of `kmerexpr.plotting.get_plot_title` for all optimizers.
    """
    problem, optimizer, initializer = exp.prob, exp.opt, exp.init

    title = (
        f"{problem.filename}"
        f"_N{problem.N}"
        f"_L{problem.L}"
        f"_K{problem.K}"
        f"_a{problem.alpha}"
    )

    if with_optimizer:
        title += (
            f"_model{problem.model_type}"
            f"_init{initializer.__class__.__name__}"
            f"_opt{exp.opt.__class__.__name__}"
            f"_maxiter{exp.opt.max_iter}"
        )

    return title


def check_all_exps_are_on_same_problem(experiments):
    exp0 = experiments[0]
    for exp in experiments:
        if not all([exp0.prob == exp.prob]):
            raise ValueError(
                f"Trying to compare experiments on different problems. "
                f"Got {exp0.prob} != {exp.prob}"
            )


@lru_cache(maxsize=2)
def load_problem_cached(prob: Problem):
    simulation_dict = prob.load_simulation_parameters()
    lengths = load_lengths(prob.filename, prob.N, prob.L)
    return simulation_dict, lengths


def nrmse(psis, psi_true):
    """Normalized Root Mean-Squared Error."""
    return [np.linalg.norm(psi - psi_true, ord=2) / np.sqrt(len(psi)) for psi in psis]


def avg_l1(psis, psi_true):
    """Average L1 norm."""
    return [np.linalg.norm(psi - psi_true, ord=1) / len(psi) for psi in psis]


def kl(psis, psi_true):
    """Average L1 norm."""
    return [entropy(psi, psi_true) for psi in psis]


def rkl(psis, psi_true):
    return [entropy(psi_true, psi) for psi in psis]


def jsd(psis, psi_true):
    avg_psis = [0.5 * psi + 0.5 * psi_true for psi in psis]
    return [
        0.5 * (entropy(psi_avg, psi) + entropy(psi_avg, psi_true))
        for psi, psi_avg in zip(psis, avg_psis)
    ]


LOSS_LABELS = {
    nrmse: "RMSE",
    avg_l1: "Avg. L1",
    kl: r"$KL(\hat\phi | \phi^*)$",
    rkl: r"$KL(\phi^* | \hat\phi)$",
    jsd: r"JSD",
}


FNAME_TEST_VS_TIME = "test-vs-time"
FNAME_TEST_VS_ITER = "test-vs-iter"
FNAME_OPTIM_VS_ITER = "optim-vs-iter"
FNAME_OPTIM_VS_TIME = "optim-vs-time"


def FNAME_ISOFORM(len_adj: bool = False, diff: bool = False):
    name = "isoform"
    if len_adj:
        name += "-length-adj"
    if diff:
        name += "-diff"
    return name


def get_opts_str(exps: List[Experiment]):
    opts = sorted(list(set([exp.opt.__class__.__name__ for exp in exps])))
    return "[" + "+".join(opts) + "]"
