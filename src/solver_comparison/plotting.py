import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kmerexpr.plotting import plot_error_vs_iterations, plot_general, plot_scatter
from kmerexpr.simulate_reads import length_adjustment_inverse
from kmerexpr.utils import Model_Parameters, load_lengths

from solver_comparison import config
from solver_comparison.experiment import Experiment
from solver_comparison.logging.expfiles import exp_filepaths
from solver_comparison.problem.problem import Problem
from solver_comparison.solvers.optimizer import Optimizer


def get_plot_base_filename(exp: Experiment):
    """Generate base filename for the experiment.

    Version of `kmerexpr.plotting.get_plot_title` for all optimizers.
    """
    problem, optimizer, initializer = exp.prob, exp.opt, exp.init
    return (
        f"{problem.filename}-{problem.model_type}-"
        f"N-{problem.N}-L-{problem.L}-K-{problem.K}-"
        f"init-{initializer.method}-a-{problem.alpha}-"
        f"{exp.opt.__class__.__name__}"
    )


def plot_against_ground_truth(dict_simulation):
    pass


def plot_optimization():
    pass


def plot_optimization_error(dict_results, title, opt_name, save_path="./figures"):
    dict_plot = {opt_name: [-np.array(dict_results["loss_records"])]}
    plot_general(
        dict_plot,
        title=title,
        save_path=save_path,
        yaxislabel=r"$f(\theta)$",
        xticks=dict_results["iteration_counts"],
        xaxislabel="iterations",
        miny=np.min(dict_plot[opt_name]),
    )
    plt.close()


def plot_stat(stat, dict_results, title, opt_name, save_path="./figures"):
    dict_plot = {opt_name: [dict_results[stat]]}
    plot_general(
        dict_plot,
        title=title + stat,
        save_path=save_path,
        yaxislabel=stat,
        xticks=dict_results["iteration_counts"],
        xaxislabel="iterations",
        miny=np.min(dict_plot[opt_name]),
    )
    plt.close()


def convert_summary_to_dict_results(summary):
    dict_results = {
        "x": summary["prob_end"],
        "xs": summary["probs"],
        "loss_records": summary["funcs"],
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


def make_individual_exp_plots(exp: Experiment):
    base_title = get_plot_base_filename(exp)
    fig_folder = config.figures_dir()

    dict_results = load_dict_result(exp)
    dict_simulation = exp.prob.load_simulation_parameters()

    theta_true = dict_simulation["theta_true"]
    psi_true = dict_simulation["psi"]
    theta_opt = dict_results["x"]
    lengths = load_lengths(exp.prob.filename, exp.prob.N, exp.prob.L)
    psi_opt = length_adjustment_inverse(theta_opt, lengths)

    plot_error_vs_iterations(
        dict_results,
        theta_true,
        base_title + "-theta-errors",
        model_type=exp.prob.model_type,
        save_path=fig_folder,
    )

    plot_optimization_error(
        dict_results,
        base_title + "-optim-errors",
        opt_name=exp.opt.__class__.__name__,
        save_path=fig_folder,
    )

    for stat in ["grads_l0", "grads_l1", "grads_l2", "grads_linf"]:
        plot_stat(
            stat,
            dict_results,
            base_title,
            opt_name=exp.prob.model_type,
            save_path=fig_folder,
        )

    plot_scatter(base_title, psi_opt, psi_true, save_path=fig_folder)
    plot_scatter(
        base_title, psi_opt, psi_opt - psi_true, horizontal=True, save_path=fig_folder
    )
