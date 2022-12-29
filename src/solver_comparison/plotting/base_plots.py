import os

import numpy as np
from kmerexpr.utils import get_errors
from matplotlib import pyplot as plt

from solver_comparison.plotting.style import LINEWIDTH, markers, palette


def save_and_close(dir_path, title):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.join(dir_path, title + ".pdf")
    plt.savefig(file_path)
    print("Saved plot ", file_path)
    plt.close()


def plot_scatter(title, xaxis, yaxis, horizontal=False, save_path="./figures"):
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


def plot_general(
    result_dict,
    title,
    save_path,
    threshold=False,
    yaxislabel=r"$ f(x^k)/f(x^0)$",
    xaxislabel="Effective Passes",
    xticks=None,
    logplot=True,
    miny=10000,
):
    for algo_name, marker, color in zip(result_dict.keys(), markers, palette):
        print("plotting: ", algo_name)
        result = result_dict[algo_name]  # result is a 2-d list with different length
        len_cut = len(
            min(result, key=len)
        )  # cut it with min_len and convert it to numpy array for plot
        result = np.array(list(map(lambda arr: arr[:len_cut], result)))
        val_avg = np.mean(result, axis=0)
        if threshold:
            len_cut = (
                np.argmax(val_avg <= threshold) + 1
                if np.sum(val_avg <= threshold) > 0
                else len(val_avg)
            )
            val_avg = val_avg[:len_cut]
        newlength = len(val_avg)
        val_min = np.min(result, axis=0)[:newlength]

        if xticks is None:
            xticks_p = np.arange(newlength)
        else:
            xticks_p = xticks[:newlength]
        markevery = 1
        if newlength > 20:
            markevery = int(np.floor(newlength / 15))
        if (
            np.min(val_avg) <= 0 or logplot == False
        ):  # this to detect negative values and prevent an error to be thrown
            plt.plot(
                xticks_p,
                val_avg,
                marker,
                markevery=markevery,
                label=algo_name,
                lw=LINEWIDTH,
                color=color,
            )
        else:
            plt.semilogy(
                xticks_p,
                val_avg,
                marker,
                markevery=markevery,
                label=algo_name,
                lw=LINEWIDTH,
                color=color,
            )

        newmincand = np.min(val_min)
        if miny > newmincand:
            miny = newmincand
    plt.ylim(bottom=miny)  # (1- (miny/np.abs(miny))*0.1)
    plt.tick_params()
    plt.legend()
    plt.xlabel(xaxislabel)
    plt.ylabel(yaxislabel)

    save_and_close(save_path, title)


def plot_error_vs_iterations(
    dict_results, theta_true, title, model_type, save_path="./figures"
):
    errors_list = []
    dict_plot = {}
    errors = get_errors(dict_results["xs"], theta_true)
    errors_list.append(errors)
    dict_plot[model_type] = errors_list
    plot_general(
        dict_plot,
        title=title,
        save_path=save_path,
        yaxislabel=r"$\|\theta -\theta^{*} \|$",
        xticks=dict_results["iteration_counts"],
        xaxislabel="iterations",
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


def plot_general_different_xaxes(
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

    plot_on_axis(
        ax,
        result_dict,
        xs_dict,
        xaxislabel,
        yaxislabel,
        logplot=logplot,
        miny=miny,
    )

    save_and_close(save_path, title)


def plot_on_axis(
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

        if np.min(result) <= 0 or logplot == False:
            ax.plot(
                xs,
                result,
                marker,
                label=algo_name,
                lw=LINEWIDTH,
                color=color,
            )
        else:
            ax.semilogy(
                xs,
                result,
                marker,
                label=algo_name,
                lw=LINEWIDTH,
                color=color,
            )

        newmincand = np.min(result)
        if miny > newmincand:
            miny = newmincand

    ax.set_ylim(bottom=miny)  # (1- (miny/np.abs(miny))*0.1)
    ax.legend()
    ax.set_xlabel(xaxislabel)
    ax.set_ylabel(yaxislabel)


def plot_multiple(
    multiple_result_dict,
    xs_dict,
    title,
    save_path,
    xaxislabel,
    logplot=True,
    miny=10000,
):
    n_plots = len(multiple_result_dict)
    fig = plt.figure()
    axes = [fig.add_subplot(1, n_plots, i) for i in range(1, n_plots + 1)]

    for i, (name, result_dict) in enumerate(multiple_result_dict.items()):
        plot_on_axis(
            axes[i],
            result_dict,
            xs_dict,
            xaxislabel,
            yaxislabel=name,
            logplot=logplot,
            miny=miny,
        )

    save_and_close(save_path, title)
