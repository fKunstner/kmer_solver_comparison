from solver_comparison.experiment import Experiment
from solver_comparison.plotting import make_individual_exp_plots
from solver_comparison.problem.problem import Problem
from solver_comparison.solvers.initializer import Initializer
from solver_comparison.solvers.optimizer import ExpGrad

experiments = [
    Experiment(
        prob=Problem(
            model_name="Simplex",
            filename="sampled_genome_0.001.fsa",
            K=8,
            N=10000,
            L=14,
            alpha=0.1,
            beta=1.0,
        ),
        opt=ExpGrad(),
        init=Initializer("simplex_uniform"),
    )
]


if __name__ == "__main__":

    for exp in experiments:
        if not exp.has_already_run():
            exp.run()

    for exp in experiments:
        make_individual_exp_plots(exp)
