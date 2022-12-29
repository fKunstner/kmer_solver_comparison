import os

import numpy as np
from kmerexpr.utils import get_errors
from matplotlib import pyplot as plt

from solver_comparison.plotting.style import markers, palette


def plot_scatter(title, xaxis, yaxis, horizontal=False, save_path="./figures"):
    plt.scatter(xaxis, yaxis, s=5, alpha=0.4)  # theta_opt
    if horizontal:
        title = title + "-psi-minus-scatter"
        plt.plot([0, np.max(xaxis)], [0, 0], "--")
        plt.ylabel(r"$ \psi^{opt} - \psi^{*}$", fontsize=25)
    else:
        max_scal = np.max([np.max(xaxis), np.max(yaxis)])
        title = title + "-psi-scatter"
        plt.plot([0, max_scal], [0, max_scal], "--")
        plt.ylabel(r"$ \psi^{*}$", fontsize=25)

    plt.xlabel(r"$ \psi^{opt}$", fontsize=25)
    plt.savefig(
        os.path.join(save_path, title + ".pdf"), bbox_inches="tight", pad_inches=0.01
    )
    print("Saved plot ", os.path.join(save_path, title + ".pdf"))
    plt.close()


def plot_general(
    result_dict,
    title,
    save_path,
    threshold=False,
    yaxislabel=r"$ f(x^k)/f(x^0)$",
    xaxislabel="Effective Passes",
    xticks=None,
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
        val_max = np.max(result, axis=0)[:newlength]
        # std_result = np.std(result, axis=0)[:newlength]
        # val_min = np.add(val_avg, -std_result)
        # val_max = np.add(val_avg, std_result)
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
                markersize=12,
                label=algo_name,
                lw=3,
                color=color,
            )
        else:
            plt.semilogy(
                xticks_p,
                val_avg,
                marker,
                markevery=markevery,
                markersize=12,
                label=algo_name,
                lw=3,
                color=color,
            )
        # plt.fill_between(xticks_p, val_min, val_max, alpha=0.2, color=color)
        newmincand = np.min(val_min)
        if miny > newmincand:
            miny = newmincand
    plt.ylim(bottom=miny)  # (1- (miny/np.abs(miny))*0.1)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=fontsize)
    plt.xlabel(xaxislabel, fontsize=25)
    plt.ylabel(yaxislabel, fontsize=25)
    # plt.title(title, fontsize=25)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(
        os.path.join(save_path, title + ".pdf"), bbox_inches="tight", pad_inches=0.01
    )
    print("Saved plot ", os.path.join(save_path, title + ".pdf"))
    return plt.gcf()  # or try plt.figure(1)


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
    yaxislabel,
    xaxislabel,
    logplot=True,
    fontsize=30,
    miny=10000,
):
    plt.rc("text", usetex=True)
    plt.rc("font", family="sans-serif")

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


def plot_on_axis(
    ax,
    result_dict,
    xs_dict,
    xaxislabel,
    yaxislabel,
    logplot=True,
    miny=100000,
    fontsize=30,
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
                markersize=12,
                label=algo_name,
                lw=3,
                color=color,
            )
        else:
            ax.semilogy(
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

    ax.set_ylim(bottom=miny)  # (1- (miny/np.abs(miny))*0.1)
    # ax.set_tick_params(labelsize=20)
    ax.legend(fontsize=fontsize)
    ax.set_xlabel(xaxislabel, fontsize=25)
    ax.set_ylabel(yaxislabel, fontsize=25)


def plot_multiple(
    multiple_result_dict,
    xs_dict,
    title,
    save_path,
    xaxislabel,
    logplot=True,
    fontsize=30,
    miny=10000,
):
    plt.rc("text", usetex=False)
    plt.rc("font", family="sans-serif")

    n_plots = len(multiple_result_dict)
    scale = 12
    fig = plt.figure(figsize=(scale * n_plots, scale))
    axes = [fig.add_subplot(1, n_plots, i) for i in range(1, n_plots + 1)]

    for i, (name, result_dict) in enumerate(multiple_result_dict.items()):
        plot_on_axis(
            axes[i],
            result_dict,
            xs_dict,
            xaxislabel,
            yaxislabel=name,
            logplot=logplot,
            fontsize=fontsize,
            miny=miny,
        )

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(
        os.path.join(save_path, title + ".pdf"), bbox_inches="tight", pad_inches=0.01
    )
    print("Saved plot ", os.path.join(save_path, title + ".pdf"))
    plt.gcf()
    plt.close()
