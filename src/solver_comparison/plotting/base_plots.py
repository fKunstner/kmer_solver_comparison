import os

import numpy as np
from kmerexpr.utils import get_errors
from matplotlib import pyplot as plt

from solver_comparison.plotting.style import (
    _GOLDEN_RATIO,
    LINEWIDTH,
    MARKERS,
    figsize,
    palette,
)


def subsample(xs, n, log=False):
    sub_idx = subsample_idx(len(xs), n=n)
    return [xs[i] for i in sub_idx]


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


def save_and_close(dir_path, title, fig=None):
    if fig is None:
        fig = plt.gcf()

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.join(dir_path, title + ".pdf")
    fig.savefig(file_path)
    print("Saved plot ", file_path)
    plt.close(fig)


##
# Making axes


def make_axis_general(
    ax,
    ys_dict,
    xs_dict,
    logplot=True,
    markers=MARKERS,
    miny=100000,
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

        newmincand = np.min(result)
        if miny > newmincand:
            miny = newmincand

    ax.set_ylim(bottom=miny)  # (1- (miny/np.abs(miny))*0.1)


##
# Making figures -- private functions / main logic


def make_figure_and_axes(
    rows, cols, height_to_width_ratio=_GOLDEN_RATIO, sharex=False, sharey=False
):
    return plt.subplots(
        nrows=rows,
        ncols=cols,
        sharex=sharex,
        sharey=sharey,
        figsize=figsize(
            nrows=rows, ncols=cols, height_to_width_ratio=height_to_width_ratio
        ),
        squeeze=False,
    )


def _make_figure_general_different_xaxes(
    ys_dict,
    xs_dict,
    title,
    save_path,
    xlabel,
    ylabel,
    logplot=True,
    miny=10000,
):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    make_axis_general(
        ax,
        ys_dict,
        xs_dict,
        logplot=logplot,
        miny=miny,
    )
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    save_and_close(save_path, title)


def _make_axis_scatter(ax, xs, ys, horizontal):
    ax.scatter(xs, ys, s=5, alpha=0.4)  # theta_opt
    if horizontal:
        ax.plot([0, np.max(xs)], [0, 0], "--")
    else:
        max_scal = np.max([np.max(xs), np.max(ys)])
        ax.plot([0, max_scal], [0, max_scal], "--")


##
# Making figures -- public api


def make_figure_scatter(title, xs, ys, horizontal=False, save_path="./figures"):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    _make_axis_scatter(ax, xs, ys, horizontal)

    if horizontal:
        title = title + "-psi-minus-scatter"
        ax.set_ylabel(r"$ \psi^{opt} - \psi^{*}$")
    else:
        title = title + "-psi-scatter"
        ax.set_ylabel(r"$ \psi^{*}$")
    plt.xlabel(r"$ \psi^{opt}$")

    save_and_close(save_path, title)


def make_figure_error_vs_iterations(
    results_dict, theta_true, title, model_type, save_path="./figures"
):
    dict_plot = {model_type: get_errors(results_dict["xs"], theta_true)}
    dict_xs = {model_type: results_dict["iteration_counts"]}
    _make_figure_general_different_xaxes(
        ys_dict=dict_plot,
        xs_dict=dict_xs,
        title=title,
        save_path=save_path,
        ylabel=r"$\|\theta -\theta^{*} \|$",
        xlabel="iterations",
    )


def make_figure_stat(stat, results_dict, title, opt_name, save_path="./figures"):
    dict_plot = {opt_name: results_dict[stat]}
    dict_xs = {opt_name: results_dict["iteration_counts"]}
    _make_figure_general_different_xaxes(
        ys_dict=dict_plot,
        xs_dict=dict_xs,
        title=title + stat,
        save_path=save_path,
        ylabel=stat,
        xlabel="iterations",
    )


def make_figure_optimization_error(
    results_dict, title, opt_name, save_path="./figures"
):
    ys_dict = {opt_name: -np.array(results_dict["loss_records"])}
    xs_dict = {opt_name: results_dict["iteration_counts"]}
    _make_figure_general_different_xaxes(
        ys_dict=ys_dict,
        xs_dict=xs_dict,
        title=title,
        save_path=save_path,
        ylabel=r"$f(\theta)$",
        xlabel="iterations",
    )


def make_figure_multiple_plots(
    multiple_result_dict,
    xs_dict,
    title,
    save_path,
    xaxislabel,
    logplot=True,
):
    n_plots = len(multiple_result_dict)
    fig = plt.figure()
    axes = [fig.add_subplot(1, n_plots, i) for i in range(1, n_plots + 1)]

    for i, (name, result_dict) in enumerate(multiple_result_dict.items()):
        make_axis_general(
            axes[i],
            result_dict,
            xs_dict,
            logplot=logplot,
        )
        axes[i].set_ylabel(name)
        axes[i].set_xlabel(xaxislabel)
    save_and_close(save_path, title)
