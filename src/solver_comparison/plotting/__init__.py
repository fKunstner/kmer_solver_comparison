import pdb
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from solver_comparison import config
from solver_comparison.experiment import Experiment
from solver_comparison.plotting.base_plots import (
    ax_xylabels,
    equalize_xy_axes,
    make_figure_and_axes,
    plot_multiple,
    plot_on_ax_convergence,
    plot_on_ax_isoform_composition,
    plot_on_axis_optimization,
    plot_on_axis_test_error,
    probability_scatter,
    save_and_close,
    subsample,
)
from solver_comparison.plotting.data import (
    FNAME_CONVERG_VS_ITER,
    FNAME_CONVERG_VS_TIME,
    FNAME_ISOFORM,
    FNAME_OPTIM_VS_ITER,
    FNAME_OPTIM_VS_TIME,
    FNAME_TEST_VS_ITER,
    FNAME_TEST_VS_TIME,
    LOSS_LABELS,
    avg_l1,
    check_all_exps_are_on_same_problem,
    fw_gap,
    get_error_at_end,
    get_opt_full_name,
    get_opts_str,
    get_plot_base_filename,
    get_plot_dirname,
    get_shortname,
    jsd,
    load_dict_result,
    load_problem_cached,
    nrmse,
    projected_grad_norm,
)
from solver_comparison.plotting.style import _GOLDEN_RATIO, base_style, figsize


def make_scatter_comparison_plots(exps: List[Experiment]):
    def __make_scatter_comparison_plots(len_adj=True, diff=False):
        fig, axes = make_figure_and_axes(
            rows=1, cols=len(exps), height_to_width_ratio=1.2, sharex=True, sharey=True
        )
        for i, exp in enumerate(exps):
            plot_on_ax_isoform_composition(axes[0][i], exp, len_adj, diff)
        return fig

    check_all_exps_are_on_same_problem(exps)
    plt.rcParams.update(base_style)
    subdir = get_plot_dirname(exps[0])
    opts = get_opts_str(exps)

    def _make_and_save(len_adj, diff):
        fig = __make_scatter_comparison_plots(len_adj=len_adj, diff=diff)
        title = f"{opts}_{FNAME_ISOFORM(len_adj=len_adj, diff=diff)}"
        save_and_close(config.figures_dir(), subdir=subdir, title=title, fig=fig)

    _make_and_save(len_adj=False, diff=False)
    _make_and_save(len_adj=False, diff=True)
    _make_and_save(len_adj=True, diff=False)
    _make_and_save(len_adj=True, diff=True)


def make_test_error_comparison_plots(exps: List[Experiment]):
    def _make_test_error_comparison(use_time=False):
        lossfunctions = [nrmse, jsd]
        fig, axes = make_figure_and_axes(
            rows=1,
            cols=len(lossfunctions),
            height_to_width_ratio=1 / 1.618,
            sharex=True,
            sharey=False,
        )
        for i, lossfunc in enumerate(lossfunctions):
            ax = axes[0][i]
            plot_on_axis_test_error(ax, exps, lossfunc, use_time=use_time)

        axes[0][0].legend()

        return fig

    check_all_exps_are_on_same_problem(exps)
    plt.rcParams.update(base_style)
    subdir = get_plot_dirname(exps[0])
    opts = get_opts_str(exps)

    def _make_and_save(use_time):
        fig = _make_test_error_comparison(use_time=use_time)
        title = f"{opts}_{FNAME_TEST_VS_TIME if use_time else FNAME_TEST_VS_ITER}"
        save_and_close(config.figures_dir(), subdir=subdir, title=title, fig=fig)

    _make_and_save(use_time=False)
    _make_and_save(use_time=True)


def make_convergence_criterion_plots(exps: List[Experiment]):
    check_all_exps_are_on_same_problem(exps)
    plt.rcParams.update(base_style)

    def _make_convergence_criterion_plots(use_time=False):
        fig, axes = make_figure_and_axes(
            rows=1,
            cols=2,
            rel_width=1.0,
            height_to_width_ratio=1 / 1.618,
            sharex=True,
            sharey=False,
        )
        plot_on_axis_optimization(axes[0][0], exps, use_time)
        plot_on_ax_convergence(axes[0][1], exps, criterion=fw_gap, use_time=use_time)
        return fig

    subdir = get_plot_dirname(exps[0])
    opts = get_opts_str(exps)

    def _make_and_save(use_time):
        fig = _make_convergence_criterion_plots(use_time=use_time)
        title = f"{opts}_{FNAME_CONVERG_VS_TIME if use_time else FNAME_CONVERG_VS_ITER}"
        save_and_close(config.figures_dir(), title=title, subdir=subdir, fig=fig)

    _make_and_save(use_time=False)
    _make_and_save(use_time=True)


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
        plot_on_axis_optimization(ax, exps, use_time)
        return fig

    subdir = get_plot_dirname(exps[0])
    opts = get_opts_str(exps)

    def _make_and_save(use_time):
        fig = _make_optim_comparison_plots(use_time=use_time)
        title = f"{opts}_{FNAME_OPTIM_VS_TIME if use_time else FNAME_OPTIM_VS_ITER}"
        save_and_close(config.figures_dir(), title=title, subdir=subdir, fig=fig)

    _make_and_save(use_time=False)
    _make_and_save(use_time=True)


def make_all_joint_plots(exps: List[Experiment]):
    make_optim_comparison_plots(exps)
    make_test_error_comparison_plots(exps)
    make_scatter_comparison_plots(exps)


def make_all_single_plots(exps: List[Experiment]):
    check_all_exps_are_on_same_problem(exps)

    subdir = get_plot_dirname(exps[0])
    opts = get_opts_str(exps)

    ##
    # Optim plots

    for use_time in [False, True]:
        fig, axes = make_figure_and_axes(1, 1, 0.5, _GOLDEN_RATIO)
        plot_on_axis_optimization(axes[0][0], exps, use_time=use_time)
        title = f"{opts}_{FNAME_OPTIM_VS_TIME if use_time else FNAME_OPTIM_VS_ITER}"
        save_and_close(config.figures_dir(), title=title, subdir=subdir, fig=fig)

    ##
    # Test plots

    for use_time in [False, True]:
        for loss_func in [nrmse, jsd]:
            fig, axes = make_figure_and_axes(1, 1, 0.5, _GOLDEN_RATIO)
            plot_on_axis_test_error(axes[0][0], exps, loss_func, use_time=use_time)
            title = (
                f"{opts}_{(FNAME_TEST_VS_TIME if use_time else FNAME_TEST_VS_ITER)}"
                + f"_{loss_func.__name__}"
            )
            save_and_close(config.figures_dir(), title=title, subdir=subdir, fig=fig)

    ##
    # Scatter plots

    for exp in exps:
        base_title = get_opt_full_name(exp)
        for length_adjusted in [True, False]:
            for horizontal in [True, False]:
                fig, axes = make_figure_and_axes(1, 1, 0.5, _GOLDEN_RATIO)
                plot_on_ax_isoform_composition(
                    axes[0][0], exp, length_adjusted, horizontal
                )
                title = base_title + f"_{FNAME_ISOFORM(length_adjusted, horizontal)}"
                save_and_close(
                    config.figures_dir(), title=title, subdir=subdir, fig=fig
                )


def make_all_plots(exps: List[Experiment]):
    make_all_single_plots(exps)
    make_all_joint_plots(exps)


def make_sensitivity_plot(exps: List[Experiment], Ks, Ls, Ns, alphas, max_iters):
    data_filename = exps[0].prob.filename

    all_on_same_data = all([exp.prob.filename == data_filename for exp in exps])
    assert all_on_same_data

    plt.rcParams.update(base_style)

    lossfunctions = [nrmse, jsd]

    def select(K, L, N, alpha, max_iter_):
        selected_exps = [
            exp
            for exp in exps
            if exp.prob.K == K
            and exp.prob.N == N
            and exp.prob.L == L
            and exp.prob.alpha == alpha
            and exp.opt.max_iter == max_iter_
        ]
        if len(selected_exps) != 1:
            import pdb

            pdb.set_trace()
        assert len(selected_exps) == 1
        return selected_exps[0]

    def _make_sensitivity_plot(max_iter, length_adjusted, lossfunc):
        fig, axes = make_figure_and_axes(
            len(Ls), len(alphas), rel_width=len(alphas) / 3
        )

        for i, L in enumerate(Ls):
            for j, alpha in enumerate(alphas):
                ax = axes[i][j]

                for K in Ks:
                    errors_vs_ns = [
                        get_error_at_end(
                            select(K=K, L=L, N=N, alpha=alpha, max_iter_=max_iter),
                            lossfunc,
                            length_adjusted=length_adjusted,
                        )
                        for N in Ns
                    ]

                    ax.plot(Ns, errors_vs_ns, "o-", label=K)

        axes[-1][-1].legend()

        for ax in [ax for col in axes for ax in col]:
            ax.set_yscale("log")
            ax.set_xscale("log")

        # ymin, ymax = np.inf, -np.inf
        # for ax in [ax for col in axes for ax in col]:
        #     ymin = min(ymin, ax.get_ylim()[0])
        #     ymax = max(ymax, ax.get_ylim()[1])
        # for ax in [ax for col in axes for ax in col]:
        #     ax.set_ylim([ymin, ymax])

        for i, L in enumerate(Ls):
            ax = axes[i][0]
            ax.set_ylabel(f"L = {L}")
        for j, alpha in enumerate(alphas):
            ax = axes[0][j]
            ax.set_title(f"$\\alpha = {alpha}$")

        if length_adjusted:
            fig.suptitle(f"{data.LOSS_LABELS[lossfunc]} (length adjusted, $\\psi$)")
        else:
            fig.suptitle(f"{data.LOSS_LABELS[lossfunc]} (raw, $\\theta$)")

    dirname = f"sensitivity_{data_filename}_Ks={Ks}_Ls={Ls}_Ns={Ns}_alphas={alphas}"

    for max_iter in max_iters:
        for lossfunc in lossfunctions:
            for length_adjusted in [True, False]:
                figname = f"metric={lossfunc.__name__}_maxiter={max_iter}_lengthadjusted={length_adjusted}"
                fig = _make_sensitivity_plot(
                    max_iter, length_adjusted=length_adjusted, lossfunc=lossfunc
                )
                save_and_close(
                    config.figures_dir(), subdir=dirname, title=figname, fig=fig
                )
