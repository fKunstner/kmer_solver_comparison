from typing import List

import matplotlib.pyplot as plt
import numpy as np
from kmerexpr.simulate_reads import length_adjustment_inverse
from kmerexpr.utils import get_errors, load_lengths
from scipy.special import rel_entr

from solver_comparison import config
from solver_comparison.experiment import Experiment
from solver_comparison.plotting import _make_figure_general_different_xaxes
from solver_comparison.plotting.base_plots import (
    _make_figure_general_different_xaxes,
    make_axis_general,
    make_figure_error_vs_iterations,
    make_figure_multiple_plots,
    make_figure_optimization_error,
    make_figure_scatter,
    make_figure_stat,
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

    results_dict = load_dict_result(exp)
    simulation_dict = exp.prob.load_simulation_parameters()

    theta_true = simulation_dict["theta_true"]
    theta_sampled = simulation_dict["theta_sampled"]
    psi_true = simulation_dict["psi"]
    theta_opt = results_dict["x"]
    lengths = load_lengths(exp.prob.filename, exp.prob.N, exp.prob.L)
    psi_opt = length_adjustment_inverse(theta_opt, lengths)

    plt.rcParams.update(figsize(ncols=1))
    make_figure_error_vs_iterations(
        results_dict,
        theta_true,
        base_title + "-theta-errors",
        model_type=exp.prob.model_type,
        save_path=fig_folder,
    )

    title = base_title + "-optim-errors"
    ys_dict = {exp.opt.__class__.__name__: -np.array(results_dict["loss_records"])}
    xs_dict = {exp.opt.__class__.__name__: results_dict["iteration_counts"]}
    _make_figure_general_different_xaxes(
        ys_dict=ys_dict,
        xs_dict=xs_dict,
        title=title,
        save_path=fig_folder,
        ylabel=r"$f(\theta)$",
        xlabel="iterations",
    )

    for stat in ["grads_l0", "grads_l1", "grads_l2", "grads_linf"]:
        make_figure_stat(
            stat,
            results_dict,
            base_title,
            opt_name=exp.prob.model_type,
            save_path=fig_folder,
        )

    make_figure_scatter(
        base_title + "-theta",
        theta_opt,
        theta_sampled,
        horizontal=False,
        save_path=fig_folder,
    )
    make_figure_scatter(
        base_title + "-theta",
        theta_opt,
        theta_opt - theta_sampled,
        horizontal=True,
        save_path=fig_folder,
    )
    make_figure_scatter(
        base_title,
        psi_opt,
        psi_true,
        save_path=fig_folder,
    )
    make_figure_scatter(
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
        results_dict = load_dict_result(exp)
        simulation_dict = exp.prob.load_simulation_parameters()
        theta_true = simulation_dict["theta_true"]

        statistics_per_optim["|theta - theta*|_1"][get_shortname(exp)] = [
            np.linalg.norm(x - theta_true, ord=1) for x in results_dict["xs"]
        ]
        statistics_per_optim["|theta - theta*|_2"][get_shortname(exp)] = [
            np.linalg.norm(x - theta_true, ord=2) for x in results_dict["xs"]
        ]
        statistics_per_optim["kl(theta|theta*)"][get_shortname(exp)] = [
            np.sum(rel_entr(x, theta_true)) for x in results_dict["xs"]
        ]
        statistics_per_optim["kl(theta*|theta)"][get_shortname(exp)] = [
            np.sum(rel_entr(x, theta_true)) for x in results_dict["xs"]
        ]
        theta_errors_per_optim[get_shortname(exp)] = get_errors(
            results_dict["xs"], theta_true
        )
        func_per_optim[get_shortname(exp)] = -np.array(results_dict["loss_records"])
        grads_l1_per_optim[get_shortname(exp)] = results_dict["grads_l1"]
        xs_dict[get_shortname(exp)] = results_dict["iteration_counts"]

    title = "compare-" + get_plot_base_filename(experiments[0], with_optimizer=False)

    plt.rcParams["figure.figsize"] = [8.0, 8.0]
    plt.rcParams["figure.dpi"] = 300

    plt.rcParams.update(
        figsize(ncols=len(statistics_per_optim), height_to_width_ratio=1.0)
    )
    make_figure_multiple_plots(
        statistics_per_optim,
        xs_dict,
        title=title + "-multiple_metrics",
        save_path=config.figures_dir(),
        xaxislabel="iterations",
    )

    plt.rcParams.update(figsize(ncols=1))
    _make_figure_general_different_xaxes(
        theta_errors_per_optim,
        xs_dict,
        title=title + "-theta",
        save_path=config.figures_dir(),
        ylabel=r"$\|\theta -\theta^* \|$",
        xlabel="iterations",
    )
    _make_figure_general_different_xaxes(
        grads_l1_per_optim,
        xs_dict,
        title=title + "-gradl1",
        save_path=config.figures_dir(),
        ylabel=r"$\|\nabla f(\theta)\|_1$",
        xlabel="iterations",
        miny=np.min([np.min(vals) for key, vals in grads_l1_per_optim.items()]),
    )
    _make_figure_general_different_xaxes(
        func_per_optim,
        xs_dict,
        title=title + "-optim-error",
        save_path=config.figures_dir(),
        ylabel=r"$f(\theta)$",
        xlabel="iterations",
        miny=np.min([np.min(vals) for key, vals in func_per_optim.items()]),
    )
