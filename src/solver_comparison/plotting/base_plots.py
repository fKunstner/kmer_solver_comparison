import os
import pdb
from typing import Callable, List

import numpy as np
from kmerexpr.simulate_reads import length_adjustment_inverse
from matplotlib import pyplot as plt
from scipy.special import softmax

from solver_comparison.experiment import Experiment
from solver_comparison.plotting.data import (
    CONVERGENCE_LABELS,
    LOSS_LABELS,
    grad_softmax_to_grad_simplex,
    jsd,
    load_dict_result,
    load_problem_cached,
    projected_grad_norm,
)
from solver_comparison.plotting.style import (
    _GOLDEN_RATIO,
    LINEWIDTH,
    MARKERS,
    figsize,
    palette,
)


def subsample_idx(length, n, log=False):
    """Returns a n-subset of [0,length-1]"""
    if log:
        log_grid = np.logspace(start=0, stop=np.log10(length - 1), num=n - 1)
        idx = [0] + list(log_grid.astype(int))
    else:
        lin_grid = np.linspace(start=0, stop=length - 1, num=n)
        idx = list(lin_grid.astype(int))
    idx = sorted(list(set(idx)))
    return idx


def subsample(xs, n, log=False):
    sub_idx = subsample_idx(len(xs), n=n)
    return [xs[i] for i in sub_idx]


def equalize_xy_axes(*axes):
    """Equalize the x and y limits to be the same.

    Ensures that ``ax.get_xlim() == ax.get_ylim()`` for each ``ax``
    """
    for ax in axes:
        axlimits = [*ax.get_xlim(), *ax.get_ylim()]
        minlim, maxlim = np.min(axlimits), np.max(axlimits)
        ax.set_xlim([minlim, maxlim])
        ax.set_ylim([minlim, maxlim])


def ax_xylabels(ax, x: str, y: str):
    ax.set_xlabel(x)
    ax.set_ylabel(y)


def save_and_close(dir_path, subdir, title, fig=None):
    if fig is None:
        fig = plt.gcf()

    if subdir is not None:
        dir_path = os.path.join(dir_path, subdir)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    fig.savefig(os.path.join(dir_path, title + ".pdf"))
    fig.savefig(os.path.join(dir_path, title + ".png"), dpi=600)
    print("Saved plot ", os.path.join(dir_path, title))
    plt.close(fig)


##
#


def make_figure_and_axes(
    rows,
    cols,
    rel_width=1.0,
    height_to_width_ratio=_GOLDEN_RATIO,
    sharex=False,
    sharey=False,
):
    return plt.subplots(
        nrows=rows,
        ncols=cols,
        sharex=sharex,
        sharey=sharey,
        figsize=figsize(
            nrows=rows,
            ncols=cols,
            rel_width=rel_width,
            height_to_width_ratio=height_to_width_ratio,
        ),
        squeeze=False,
    )


def plot_multiple(
    ax,
    ys_dict,
    xs_dict,
    markers=MARKERS,
    logplot=True,
):
    for algo_name, marker, color in zip(ys_dict.keys(), markers, palette):
        result = ys_dict[algo_name]
        xs = xs_dict[algo_name]

        ax.plot(
            xs,
            result,
            marker,
            label=algo_name,
            lw=LINEWIDTH,
            color=color,
        )
        if logplot and not (np.min(result) <= 0):
            ax.set_yscale("log")


def probability_scatter(ax, xs, ys, horizontal):
    ax.scatter(xs, ys, s=5, alpha=0.4)  # theta_opt
    if horizontal:
        ax.plot([0, np.max(xs)], [0, 0], "--")
    else:
        max_scal = np.max([np.max(xs), np.max(ys)])
        ax.plot([0, max_scal], [0, max_scal], "--")


##
#


def plot_on_ax_convergence(ax, exps: List[Experiment], criterion, use_time: bool):
    xs_dict = {}
    ys_dict = {}

    for exp in exps:
        opt_name = exp.opt.__class__.__name__
        results_dict = load_dict_result(exp)
        params = results_dict["xs"]
        grads = results_dict["grads"]

        if exp.prob.model_type == "Softmax":
            params = [np.array(softmax(param)) for param in params]
            grads = [
                grad_softmax_to_grad_simplex(param, grad)
                for param, grad in zip(params, grads)
            ]

        xs = results_dict["times"] if use_time else results_dict["iteration_counts"]
        ys = [criterion(param, grad) for param, grad in zip(params, grads)]

        xs_dict[opt_name] = subsample(xs, n=50)
        ys_dict[opt_name] = subsample(ys, n=50)

    plot_multiple(
        ax,
        xs_dict=xs_dict,
        ys_dict=ys_dict,
        markers=[""] * len(exps),
        logplot=True,
    )

    ax.set_xlabel("Time" if use_time else "Iteration")
    ax.set_title(CONVERGENCE_LABELS[criterion])


def plot_on_ax_isoform_composition(
    ax, exp: Experiment, length_adjusted: bool = True, horizontal: bool = False
):
    if not length_adjusted:
        name = "Isoform Composition"
        varname = "theta"
    else:
        name = "Length-adjusted"
        varname = "psi"
    if horizontal:
        xlabel = rf"$\{varname}^*$"
        ylabel = rf"$\{varname}^* - \hat\{varname}$"
    else:
        xlabel = rf"$\hat\{varname}$"
        ylabel = rf"$\{varname}^*$"

    simulation_dict, lengths = load_problem_cached(exp.prob)
    results_dict = load_dict_result(exp)
    theta_opt = results_dict["x"]
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

    probability_scatter(ax, x, y, horizontal=horizontal)

    if not horizontal:
        equalize_xy_axes(ax)

    ax.set_title(exp.opt.__class__.__name__)
    ax.set_xlabel(xlabel)

    ax.set_ylabel(f"{name}\n{ylabel}")


def plot_on_axis_test_error(
    ax, exps: List[Experiment], loss_func: Callable, use_time=False
):
    xs_dict = {}
    ys_dict = {}

    simulation_dict, lengths = load_problem_cached(exps[0].prob)

    for exp in exps:
        results_dict = load_dict_result(exp)
        learned_isoform_compositions = [
            length_adjustment_inverse(x, lengths) for x in results_dict["xs"]
        ]
        true_isoform_composition = simulation_dict["psi"]
        opt_name = exp.opt.__class__.__name__

        if use_time:
            xs = results_dict["times"]
        else:
            xs = results_dict["iteration_counts"]

        #        if loss_func == jsd and exp.opt.__class__.__name__ == "MG":
        #            pdb.set_trace()

        ys = loss_func(learned_isoform_compositions, true_isoform_composition)

        xs_dict[opt_name] = subsample(xs, n=50)
        ys_dict[opt_name] = subsample(ys, n=50)

    plot_multiple(
        ax,
        xs_dict=xs_dict,
        ys_dict=ys_dict,
        markers=[""] * len(exps),
        logplot=True,
    )

    ax.set_title(LOSS_LABELS[loss_func])

    if use_time:
        ax.set_xscale("log")
        ax.set_xlabel("Time")
    else:
        ax.set_xlabel("Iteration")

    everything_is_inf = np.all([np.isinf(y) for line in ax.lines for y in line._y])
    if everything_is_inf:
        ax.text(0, 0, "Everything is INF")
        ax.set_xscale("linear")

    ax.legend()


def plot_on_axis_optimization(ax, exps: List[Experiment], use_time: bool = False):
    xs_dict = {}
    ys_dict = {}
    for exp in exps:
        results_dict = load_dict_result(exp)
        ys = -np.array(results_dict["objs"])

        if use_time:
            xs = results_dict["times"]
        else:
            xs = results_dict["iteration_counts"]

        opt_name = exp.opt.__class__.__name__
        xs_dict[opt_name] = xs
        ys_dict[opt_name] = ys

    plot_multiple(
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
