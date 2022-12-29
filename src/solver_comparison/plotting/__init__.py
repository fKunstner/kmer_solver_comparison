from typing import List

import numpy as np
from kmerexpr.simulate_reads import length_adjustment_inverse
from kmerexpr.utils import get_errors, load_lengths
from matplotlib import pyplot as plt
from scipy.special import rel_entr
from scipy.stats import entropy

from solver_comparison import config
from solver_comparison.experiment import Experiment
from solver_comparison.plotting.base_plots import (
    _make_axis_scatter,
    _make_figure_general_different_xaxes,
    ax_xylabels,
    equalize_xy_axes,
    make_axis_general,
    make_figure_and_axes,
    make_figure_error_vs_iterations,
    make_figure_multiple_plots,
    make_figure_optimization_error,
    make_figure_scatter,
    make_figure_stat,
    save_and_close,
    subsample,
    subsample_idx,
)
from solver_comparison.plotting.data import (
    check_all_exps_are_on_same_problem,
    get_plot_base_filename,
    get_shortname,
    load_dict_result,
)
from solver_comparison.plotting.style import _GOLDEN_RATIO, base_style, figsize


def make_scatter_comparison_plots(exps: List[Experiment]):
    check_all_exps_are_on_same_problem(exps)

    plt.rcParams.update(base_style)

    def _make_scatter_comparison_plots(length_adjusted=True, horizontal=False):
        fig, axes = make_figure_and_axes(
            rows=1, cols=len(exps), height_to_width_ratio=1.2, sharex=True, sharey=True
        )

        if not length_adjusted:
            name = "Isoform Composition"
            varname = "psi"
        else:
            name = "Length-adjusted"
            varname = "theta"
        if horizontal:
            xlabel = rf"$\{varname}^*$"
            ylabel = rf"$\{varname}^* - \hat\{varname}$"
        else:
            xlabel = rf"$\hat\{varname}$"
            ylabel = rf"$\{varname}^*$"

        for i, exp in enumerate(exps):
            ax = axes[0][i]

            results_dict = load_dict_result(exp)
            simulation_dict = exp.prob.load_simulation_parameters()

            theta_opt = results_dict["x"]
            lengths = load_lengths(exp.prob.filename, exp.prob.N, exp.prob.L)
            psi_opt = length_adjustment_inverse(theta_opt, lengths)

            psi_true = simulation_dict["psi"]
            theta_true = simulation_dict["theta_true"]

            if length_adjusted:
                true = psi_true
                est = psi_opt
            else:
                true = theta_true
                est = theta_opt

            if horizontal:
                x = true
                y = true - est
            else:
                x = est
                y = true

            _make_axis_scatter(ax, x, y, horizontal=horizontal)

            if not horizontal:
                equalize_xy_axes(ax)

            ax.set_title(exp.opt.__class__.__name__)
            ax.set_xlabel(xlabel)

        axes[0][0].set_ylabel(f"{name}\n{ylabel}")
        return fig

    base_title = get_plot_base_filename(exps[0], with_optimizer=False)

    fig = _make_scatter_comparison_plots(length_adjusted=False, horizontal=False)
    title = base_title + "-isoform"
    save_and_close(config.figures_dir(), title=title, fig=fig)

    fig = _make_scatter_comparison_plots(length_adjusted=False, horizontal=True)
    title = base_title + "-isoform-difference"
    save_and_close(config.figures_dir(), title=title, fig=fig)

    fig = _make_scatter_comparison_plots(length_adjusted=True, horizontal=False)
    title = base_title + "-lengthadjusted"
    save_and_close(config.figures_dir(), title=title, fig=fig)

    fig = _make_scatter_comparison_plots(length_adjusted=True, horizontal=True)
    title = base_title + "-lengthadjusted-difference"
    save_and_close(config.figures_dir(), title=title, fig=fig)


def make_test_error_comparison_plots(exps: List[Experiment]):
    check_all_exps_are_on_same_problem(exps)

    plt.rcParams.update(base_style)

    def _make_test_error_comparison(use_time=False):
        fig, axes = make_figure_and_axes(
            rows=1, cols=len(exps), height_to_width_ratio=1.2, sharex=True, sharey=False
        )

        def rmse(psis, psi_true):
            return [
                np.linalg.norm(psi - psi_true, ord=2) / np.sqrt(len(psi))
                for psi in psis
            ]

        def l1(psis, psi_true):
            return [np.linalg.norm(psi - psi_true, ord=1) / len(psi) for psi in psis]

        def kl(psis, psi_true):
            return [entropy(psi, psi_true) for psi in psis]

        def rkl(psis, psi_true):
            return [entropy(psi_true, psi) for psi in psis]

        lossfunctions = {
            "RMSE": rmse,
            "Avg. L1": l1,
            r"$KL(\hat\phi | \phi^*)$": kl,
            r"$KL(\phi^* | \hat\phi)$": rkl,
        }

        for i, (name, lossfunc) in enumerate(lossfunctions.items()):
            ax = axes[0][i]

            xs_dict = {}
            ys_dict = {}

            for exp in exps:
                results_dict = load_dict_result(exp)
                lengths = load_lengths(exp.prob.filename, exp.prob.N, exp.prob.L)
                isoform_compositions = [
                    length_adjustment_inverse(x, lengths) for x in results_dict["xs"]
                ]

                simulation_dict = exp.prob.load_simulation_parameters()
                psi_true = simulation_dict["psi"]
                opt_name = exp.opt.__class__.__name__

                if use_time:
                    xs = results_dict["times"]
                else:
                    xs = results_dict["iteration_counts"]

                ys = lossfunc(isoform_compositions, psi_true)

                xs_dict[opt_name] = subsample(xs, n=50)
                ys_dict[opt_name] = subsample(ys, n=50)

            make_axis_general(
                ax,
                xs_dict=xs_dict,
                ys_dict=ys_dict,
                markers=[""] * len(exps),
                logplot=True,
            )

            ax.set_title(name)
            ax.set_xlabel("Time" if use_time else "Iteration")

            if use_time:
                ax.set_xscale("log")

        axes[0][0].legend()

        return fig

    base_title = get_plot_base_filename(exps[0], with_optimizer=False)

    fig = _make_test_error_comparison(use_time=False)
    title = base_title + "-test-error-vs-iter"
    save_and_close(config.figures_dir(), title=title, fig=fig)

    fig = _make_test_error_comparison(use_time=True)
    title = base_title + "-test-error-vs-time"
    save_and_close(config.figures_dir(), title=title, fig=fig)


def make_optim_comparison_plots(exps: List[Experiment]):
    check_all_exps_are_on_same_problem(exps)

    plt.rcParams.update(base_style)

    def _make_optim_comparison_plots(use_time=False):
        fig, axes = make_figure_and_axes(
            rows=1,
            cols=1,
            rel_width=0.5,
            height_to_width_ratio=_GOLDEN_RATIO,
            sharex=True,
            sharey=False,
        )

        ax = axes[0][0]

        xs_dict = {}
        ys_dict = {}

        for exp in exps:
            results_dict = load_dict_result(exp)
            ys = -np.array(results_dict["loss_records"])

            if use_time:
                xs = results_dict["times"]
            else:
                xs = results_dict["iteration_counts"]

            opt_name = exp.opt.__class__.__name__
            xs_dict[opt_name] = xs
            ys_dict[opt_name] = ys

        make_axis_general(
            ax,
            xs_dict=xs_dict,
            ys_dict=ys_dict,
            # markers=[""] * len(exps),
            logplot=True,
        )

        ax.set_ylabel("Loss")
        ax.set_xlabel("Time" if use_time else "Iteration")

        if use_time:
            ax.set_xscale("log")

        ax.legend()

        return fig

    fig = _make_optim_comparison_plots(use_time=False)
    base_title = get_plot_base_filename(exps[0], with_optimizer=False)
    title = base_title + "-optim-vs-iter"
    save_and_close(config.figures_dir(), title=title, fig=fig)

    fig = _make_optim_comparison_plots(use_time=True)
    base_title = get_plot_base_filename(exps[0], with_optimizer=False)
    title = base_title + "-optim-vs-time"
    save_and_close(config.figures_dir(), title=title, fig=fig)


def make_all_plots(exps: List[Experiment]):
    make_optim_comparison_plots(exps)
    # make_test_error_comparison_plots(exps)
    # make_scatter_comparison_plots(exps)
