from typing import List

import matplotlib.pyplot as plt
import numpy as np
from kmerexpr.simulate_reads import length_adjustment_inverse
from kmerexpr.utils import get_errors, load_lengths
from scipy.special import rel_entr

from solver_comparison import config
from solver_comparison.experiment import Experiment
from solver_comparison.plotting.base_plots import (
    plot_error_vs_iterations,
    plot_general,
    plot_general_different_xaxes,
    plot_multiple,
    plot_on_axis,
    plot_optimization_error,
    plot_scatter,
    plot_stat,
)
from solver_comparison.plotting.data import (
    get_plot_base_filename,
    get_shortname,
    load_dict_result,
)
from solver_comparison.plotting.style import base_style, figsize


def make_individual_exp_plots(exp: Experiment):
    plt.rcParams.update(base_style)

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

    plt.rcParams.update(figsize(ncols=1))
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


def make_comparison_plots(experiments: List[Experiment]):
    plt.rcParams.update(base_style)

    theta_errors_per_optim = {}
    func_per_optim = {}
    grads_l1_per_optim = {}
    xs_dict = {}
    statistics_per_optim = {
        "|theta - theta*|_1": {},
        "|theta - theta*|_2": {},
        "kl(theta|theta*)": {},
        "kl(theta*|theta)": {},
    }

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

        statistics_per_optim["|theta - theta*|_1"][get_shortname(exp)] = [
            np.linalg.norm(x - theta_true, ord=1) for x in dict_results["xs"]
        ]
        statistics_per_optim["|theta - theta*|_2"][get_shortname(exp)] = [
            np.linalg.norm(x - theta_true, ord=2) for x in dict_results["xs"]
        ]
        statistics_per_optim["kl(theta|theta*)"][get_shortname(exp)] = [
            np.sum(rel_entr(x, theta_true)) for x in dict_results["xs"]
        ]
        statistics_per_optim["kl(theta*|theta)"][get_shortname(exp)] = [
            np.sum(rel_entr(x, theta_true)) for x in dict_results["xs"]
        ]
        theta_errors_per_optim[get_shortname(exp)] = get_errors(
            dict_results["xs"], theta_true
        )
        func_per_optim[get_shortname(exp)] = -np.array(dict_results["loss_records"])
        grads_l1_per_optim[get_shortname(exp)] = dict_results["grads_l1"]
        xs_dict[get_shortname(exp)] = dict_results["iteration_counts"]

    title = "compare-" + get_plot_base_filename(experiments[0], with_optimizer=False)

    plt.rcParams["figure.figsize"] = [8.0, 8.0]
    plt.rcParams["figure.dpi"] = 300

    plt.rcParams.update(
        figsize(ncols=len(statistics_per_optim), height_to_width_ratio=1.0)
    )
    plot_multiple(
        statistics_per_optim,
        xs_dict,
        title=title + "-multiple_metrics",
        save_path=config.figures_dir(),
        xaxislabel="iterations",
    )

    plt.rcParams.update(figsize(ncols=1))
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
