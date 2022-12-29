import os

import numpy as np
from kmerexpr.utils import get_errors
from matplotlib import pyplot as plt

from solver_comparison.plotting.style import LINEWIDTH, markers, palette


def ax_title_xy(ax, title: str, x: str, y: str):
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)


def save_and_close(dir_path, title):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.join(dir_path, title + ".pdf")
    plt.savefig(file_path)
    print("Saved plot ", file_path)
    plt.close()


##
# Making axes


def make_axis_general(
    ax,
    result_dict,
    xs_dict,
    xaxislabel,
    yaxislabel,
    logplot=True,
    miny=100000,
):
    for algo_name, marker, color in zip(result_dict.keys(), markers, palette):
        print("plotting: ", algo_name)
        result = result_dict[algo_name]
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
    ax.legend()
    ax.set_xlabel(xaxislabel)
    ax.set_ylabel(yaxislabel)


##
# Making figures -- private functions / main logic


def _make_figure_general_different_xaxes(
    result_dict,
    xs_dict,
    title,
    save_path,
    xaxislabel,
    yaxislabel,
    logplot=True,
    miny=10000,
):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    make_axis_general(
        ax,
        result_dict,
        xs_dict,
        xaxislabel,
        yaxislabel,
        logplot=logplot,
        miny=miny,
    )

    save_and_close(save_path, title)


##
# Making figures -- public api


def make_figure_scatter(title, xaxis, yaxis, horizontal=False, save_path="./figures"):
    plt.scatter(xaxis, yaxis, s=5, alpha=0.4)  # theta_opt
    if horizontal:
        title = title + "-psi-minus-scatter"
        plt.plot([0, np.max(xaxis)], [0, 0], "--")
        plt.ylabel(r"$ \psi^{opt} - \psi^{*}$")
    else:
        max_scal = np.max([np.max(xaxis), np.max(yaxis)])
        title = title + "-psi-scatter"
        plt.plot([0, max_scal], [0, max_scal], "--")
        plt.ylabel(r"$ \psi^{*}$")

    plt.xlabel(r"$ \psi^{opt}$")
    save_and_close(save_path, title)


def make_figure_error_vs_iterations(
    dict_results, theta_true, title, model_type, save_path="./figures"
):
    dict_plot = {model_type: get_errors(dict_results["xs"], theta_true)}
    dict_xs = {model_type: dict_results["iteration_counts"]}
    _make_figure_general_different_xaxes(
        result_dict=dict_plot,
        xs_dict=dict_xs,
        title=title,
        save_path=save_path,
        yaxislabel=r"$\|\theta -\theta^{*} \|$",
        xaxislabel="iterations",
    )


def make_figure_stat(stat, dict_results, title, opt_name, save_path="./figures"):
    dict_plot = {opt_name: dict_results[stat]}
    dict_xs = {opt_name: dict_results["iteration_counts"]}
    _make_figure_general_different_xaxes(
        result_dict=dict_plot,
        xs_dict=dict_xs,
        title=title + stat,
        save_path=save_path,
        yaxislabel=stat,
        xaxislabel="iterations",
    )


def make_figure_optimization_error(
    dict_results, title, opt_name, save_path="./figures"
):
    dict_plot = {opt_name: -np.array(dict_results["loss_records"])}
    dict_xs = {opt_name: dict_results["iteration_counts"]}
    _make_figure_general_different_xaxes(
        result_dict=dict_plot,
        xs_dict=dict_xs,
        title=title,
        save_path=save_path,
        yaxislabel=r"$f(\theta)$",
        xaxislabel="iterations",
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
            xaxislabel,
            yaxislabel=name,
            logplot=logplot,
        )

    save_and_close(save_path, title)
