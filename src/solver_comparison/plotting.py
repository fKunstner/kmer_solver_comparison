import json
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kmerexpr.plotting import plot_error_vs_iterations, plot_general, plot_scatter
from kmerexpr.simulate_reads import length_adjustment_inverse
from kmerexpr.utils import Model_Parameters, get_errors, load_lengths

from solver_comparison import config
from solver_comparison.experiment import Experiment
from solver_comparison.logging.expfiles import exp_filepaths
from solver_comparison.problem.problem import Problem
from solver_comparison.solvers.optimizer import Optimizer


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
        title += f"-init-{initializer.method}-{exp.opt.__class__.__name__}"

    return title


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
    theta_sampled = dict_simulation["theta_sampled"]
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

    plot_scatter(
        base_title + "-theta",
        theta_opt,
        theta_sampled,
        horizontal=False,
        save_path=fig_folder,
    )
    plot_scatter(
        base_title + "-theta",
        theta_opt,
        theta_opt - theta_sampled,
        horizontal=True,
        save_path=fig_folder,
    )
    plot_scatter(
        base_title,
        psi_opt,
        psi_true,
        save_path=fig_folder,
    )
    plot_scatter(
        base_title,
        psi_opt,
        psi_opt - psi_true,
        horizontal=True,
        save_path=fig_folder,
    )


def plot_general_different_xaxes(
    result_dict,
    xs_dict,
    title,
    save_path,
    yaxislabel,
    xaxislabel,
    logplot=True,
    fontsize=30,
    miny=10000,
):
    plt.rc("text", usetex=True)
    plt.rc("font", family="sans-serif")
    palette = [
        "#377eb8",
        "#ff7f00",
        "#984ea3",
        "#4daf4a",
        "#e41a1c",
        "brown",
        "green",
        "red",
    ]
    markers = [
        "^-",
        "1-",
        "*-",
        "s-",
        "+-",
        "o-",
        ">-",
        "d-",
        "2-",
        "3-",
        "4-",
        "8-",
        "<-",
    ]

    for algo_name, marker, color in zip(result_dict.keys(), markers, palette):
        print("plotting: ", algo_name)
        result = result_dict[algo_name]
        xs = xs_dict[algo_name]

        if np.min(result) <= 0 or logplot == False:
            plt.plot(
                xs,
                result,
                marker,
                markersize=12,
                label=algo_name,
                lw=3,
                color=color,
            )
        else:
            plt.semilogy(
                xs,
                result,
                marker,
                markersize=12,
                label=algo_name,
                lw=3,
                color=color,
            )

        newmincand = np.min(result)
        if miny > newmincand:
            miny = newmincand

    plt.ylim(bottom=miny)  # (1- (miny/np.abs(miny))*0.1)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=fontsize)
    plt.xlabel(xaxislabel, fontsize=25)
    plt.ylabel(yaxislabel, fontsize=25)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(
        os.path.join(save_path, title + ".pdf"), bbox_inches="tight", pad_inches=0.01
    )
    print("Saved plot ", os.path.join(save_path, title + ".pdf"))
    plt.gcf()
    plt.close()


def make_comparison_plots(experiments: List[Experiment]):
    theta_errors_per_optim = {}
    func_per_optim = {}
    grads_l1_per_optim = {}
    xs_dict = {}

    exp0 = experiments[0]
    for exp in experiments:
        if not all(
            [
                exp0.prob.filename == exp.prob.filename,
                exp0.prob.L == exp.prob.L,
                exp0.prob.N == exp.prob.N,
                exp0.prob.K == exp.prob.K,
                exp0.prob.alpha == exp.prob.alpha,
            ]
        ):
            raise ValueError(
                f"Trying to compare experiments on different problems. "
                f"Got {exp0.prob} != {exp.prob}"
            )

    for exp in experiments:
        dict_results = load_dict_result(exp)
        dict_simulation = exp.prob.load_simulation_parameters()
        theta_true = dict_simulation["theta_true"]
        theta_errors_per_optim[get_shortname(exp)] = get_errors(
            dict_results["xs"], theta_true
        )
        func_per_optim[get_shortname(exp)] = -np.array(dict_results["loss_records"])
        grads_l1_per_optim[get_shortname(exp)] = dict_results["grads_l1"]
        xs_dict[get_shortname(exp)] = dict_results["iteration_counts"]

    title = "compare-" + get_plot_base_filename(experiments[0], with_optimizer=False)

    plot_general_different_xaxes(
        theta_errors_per_optim,
        xs_dict,
        title=title + "-theta",
        save_path=config.figures_dir(),
        yaxislabel=r"$\|\theta -\theta^* \|$",
        xaxislabel="iterations",
    )
    plot_general_different_xaxes(
        grads_l1_per_optim,
        xs_dict,
        title=title + "-gradl1",
        save_path=config.figures_dir(),
        yaxislabel=r"$\|\nabla f(\theta)\|_1$",
        xaxislabel="iterations",
        miny=np.min([np.min(vals) for key, vals in grads_l1_per_optim.items()]),
    )
    plot_general_different_xaxes(
        func_per_optim,
        xs_dict,
        title=title + "-optim-error",
        save_path=config.figures_dir(),
        yaxislabel=r"$f(\theta)$",
        xaxislabel="iterations",
        miny=np.min([np.min(vals) for key, vals in func_per_optim.items()]),
    )
