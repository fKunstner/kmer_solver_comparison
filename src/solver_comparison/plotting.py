import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from kmerexpr.plotting import plot_error_vs_iterations, plot_scatter
from kmerexpr.simulate_reads import length_adjustment_inverse
from kmerexpr.utils import Model_Parameters, get_plot_title, load_lengths
from solver_comparison import config
from solver_comparison.experiment import Experiment
from solver_comparison.logging.expfiles import exp_filepaths


def plot_against_ground_truth(dict_simulation):
    pass


def plot_optimization():
    pass


def convert_summary_to_dict_results(summary_df):
    summary_dict = {k: v[0] for k, v in summary_df.to_dict().items()}
    dict_results = {
        "x": np.array(json.loads(summary_dict["x"])),
        "loss_records": summary_dict["loss_records"],
        "iteration_counts": summary_dict["iteration_counts"],
        "grad": np.array(json.loads(summary_dict["grad"])),
    }
    return dict_results


def make_individual_exp_plots(exp: Experiment):
    model_parameters = Model_Parameters(
        model_type=exp.prob.model_name,
        solver_name=exp.opt.solver_name,
    )

    # This is a hacky workaround.
    """Model_Parameters is used in KmerExpr to define the optimizer to use
    and define its name for plotting. If the solver_name is unknown, it defaults
    to exp_grad. We're trying to use the plotting code without modifying it,
    but need but we need to set our optimizers name"""
    model_parameters.solver_name = exp.opt.solver_name

    problem = exp.prob.kmer_problem
    conf_path, data_path, summary_path = exp_filepaths(exp.hash())
    summary_df = pd.read_csv(summary_path)
    dict_results = convert_summary_to_dict_results(summary_df)

    base_title = get_plot_title(problem, model_parameters)

    # Plotting and checking against ground truth
    dict_simulation = exp.prob.load_simulation_parameters()
    theta_true = dict_simulation["theta_true"]
    psi_true = dict_simulation["psi"]

    fig_folder = os.path.join(config.workspace(), "figures")
    Path(fig_folder).mkdir(parents=True, exist_ok=True)

    if model_parameters.model_type == "simplex":
        title_errors = base_title + "-theta-errors-"
        plot_error_vs_iterations(
            dict_results,
            theta_true,
            title_errors,
            model_type="simplex",
            save_path=fig_folder,
        )

    # Plotting scatter of theta_{opt} vs theta_{*} for a fixed k
    theta_opt = dict_results["x"]
    lengths = load_lengths(problem.filename, problem.N, problem.L)
    psi_opt = length_adjustment_inverse(theta_opt, lengths)

    plot_scatter(base_title, psi_opt, psi_true, save_path=fig_folder)
    plot_scatter(
        base_title, psi_opt, psi_opt - psi_true, horizontal=True, save_path=fig_folder
    )
