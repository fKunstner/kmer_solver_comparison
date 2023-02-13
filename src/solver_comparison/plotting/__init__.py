from typing import List

from matplotlib import pyplot as plt

from solver_comparison import config
from solver_comparison.experiment import Experiment
from solver_comparison.plotting.base_plots import (
    ax_xylabels,
    equalize_xy_axes,
    make_figure_and_axes,
    plot_multiple,
    plot_on_ax_isoform_composition,
    plot_on_axis_optimization,
    plot_on_axis_test_error,
    probability_scatter,
    save_and_close,
    subsample,
)
from solver_comparison.plotting.data import (
    FNAME_ISOFORM,
    FNAME_OPTIM_VS_ITER,
    FNAME_OPTIM_VS_TIME,
    FNAME_TEST_VS_ITER,
    FNAME_TEST_VS_TIME,
    LOSS_LABELS,
    avg_l1,
    check_all_exps_are_on_same_problem,
    get_plot_base_filename,
    get_shortname,
    kl,
    load_dict_result,
    load_problem_cached,
    nrmse,
    rkl,
)
from solver_comparison.plotting.style import _GOLDEN_RATIO, base_style, figsize


def make_scatter_comparison_plots(exps: List[Experiment]):
    def __make_scatter_comparison_plots(length_adjusted=True, horizontal=False):
        fig, axes = make_figure_and_axes(
            rows=1, cols=len(exps), height_to_width_ratio=1.2, sharex=True, sharey=True
        )
        for i, exp in enumerate(exps):
            plot_on_ax_isoform_composition(axes[0][i], exp, length_adjusted, horizontal)
        return fig

    check_all_exps_are_on_same_problem(exps)
    plt.rcParams.update(base_style)
    base_title = get_plot_base_filename(exps[0], with_optimizer=False)

    fig = __make_scatter_comparison_plots(length_adjusted=False, horizontal=False)
    title = f"{base_title}_ALL_{FNAME_ISOFORM(length_adjusted=False, horizontal=False)}"
    save_and_close(config.figures_dir(), title=title, fig=fig)

    fig = __make_scatter_comparison_plots(length_adjusted=False, horizontal=True)
    title = f"{base_title}_ALL_{FNAME_ISOFORM(length_adjusted=False, horizontal=True)}"
    save_and_close(config.figures_dir(), title=title, fig=fig)

    fig = __make_scatter_comparison_plots(length_adjusted=True, horizontal=False)
    title = f"{base_title}_ALL_{FNAME_ISOFORM(length_adjusted=True, horizontal=False)}"
    save_and_close(config.figures_dir(), title=title, fig=fig)

    fig = __make_scatter_comparison_plots(length_adjusted=True, horizontal=True)
    title = f"{base_title}_ALL_{FNAME_ISOFORM(length_adjusted=True, horizontal=True)}"
    save_and_close(config.figures_dir(), title=title, fig=fig)


def make_test_error_comparison_plots(exps: List[Experiment]):
    def _make_test_error_comparison(use_time=False):
        lossfunctions = [nrmse, avg_l1,  rkl]  #kl,
        fig, axes = make_figure_and_axes(
            rows=1, cols=len(lossfunctions), height_to_width_ratio=1.2, sharex=True, sharey=False
        )
        for i, lossfunc in enumerate(lossfunctions):
            ax = axes[0][i]
            plot_on_axis_test_error(ax, exps, lossfunc, use_time=use_time)

        axes[0][0].legend()

        return fig

    check_all_exps_are_on_same_problem(exps)
    plt.rcParams.update(base_style)
    base_title = get_plot_base_filename(exps[0], with_optimizer=False)


    fig = _make_test_error_comparison(use_time=False)
    title = f"{base_title}_ALL_{FNAME_TEST_VS_ITER}"
    save_and_close(config.figures_dir(), title=title, fig=fig)

    fig = _make_test_error_comparison(use_time=True)
    title = f"{base_title}_ALL_{FNAME_TEST_VS_TIME}"
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
        plot_on_axis_optimization(ax, exps, use_time)
        return fig

    fig = _make_optim_comparison_plots(use_time=False)
    base_title = get_plot_base_filename(exps[0], with_optimizer=False)
    title = f"{base_title}_ALL_{FNAME_OPTIM_VS_ITER}"
    save_and_close(config.figures_dir(), title=title, fig=fig)

    fig = _make_optim_comparison_plots(use_time=True)
    base_title = get_plot_base_filename(exps[0], with_optimizer=False)
    title = f"{base_title}_ALL_{FNAME_OPTIM_VS_TIME}"
    save_and_close(config.figures_dir(), title=title, fig=fig)


def make_all_joint_plots(exps: List[Experiment]):
    make_optim_comparison_plots(exps)
    make_test_error_comparison_plots(exps)
    make_scatter_comparison_plots(exps)


def make_all_single_plots(exps: List[Experiment]):
    check_all_exps_are_on_same_problem(exps)

    ##
    # Optim plots

    base_title = get_plot_base_filename(exps[0], with_optimizer=False)
    for use_time in [False, True]:
        fig, axes = make_figure_and_axes(1, 1, 0.5, _GOLDEN_RATIO)
        plot_on_axis_optimization(axes[0][0], exps, use_time=use_time)
        title = (
            f"{base_title}_{FNAME_OPTIM_VS_TIME if use_time else FNAME_OPTIM_VS_ITER}"
        )
        save_and_close(config.figures_dir(), title=title, fig=fig)

    ##
    # Test plots

    base_title = get_plot_base_filename(exps[0], with_optimizer=False)
    for use_time in [False, True]:
        for loss_func in [nrmse, avg_l1, kl, rkl]:
            fig, axes = make_figure_and_axes(1, 1, 0.5, _GOLDEN_RATIO)
            plot_on_axis_test_error(axes[0][0], exps, loss_func, use_time=use_time)
            title = (
                base_title
                + f"_{(FNAME_TEST_VS_TIME if use_time else FNAME_TEST_VS_ITER)}"
                + f"_{loss_func.__name__}"
            )
            save_and_close(config.figures_dir(), title=title, fig=fig)

    ##
    # Scatter plots

    for exp in exps:
        base_title = get_plot_base_filename(exp, with_optimizer=True)
        for length_adjusted in [True, False]:
            for horizontal in [True, False]:
                fig, axes = make_figure_and_axes(1, 1, 0.5, _GOLDEN_RATIO)
                plot_on_ax_isoform_composition(
                    axes[0][0], exp, length_adjusted, horizontal
                )
                title = base_title + f"_{FNAME_ISOFORM(length_adjusted, horizontal)}"
                save_and_close(config.figures_dir(), title=title, fig=fig)


def make_all_plots(exps: List[Experiment]):
    make_all_single_plots(exps)
    make_all_joint_plots(exps)
