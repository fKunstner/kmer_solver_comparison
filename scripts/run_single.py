from solver_comparison.experiment import Experiment
from solver_comparison.plotting import make_individual_exp_plots
from solver_comparison.problem.model import Softmax
from solver_comparison.problem.problem import Problem
from solver_comparison.solvers.expgrad import ExpGrad
from solver_comparison.solvers.initializer import Initializer
from solver_comparison.solvers.lbfgs import LBFGS

experiments = [
    Experiment(
        prob=Problem(
            model_type=Softmax,
            filename="sampled_genome_0.01.fsa",
            K=8,
            N=100000,
            L=14,
            alpha=0.1,
            beta=1.0,
        ),
        opt=LBFGS(),
        init=Initializer("simplex_uniform"),
    )
]


if __name__ == "__main__":

    for exp in experiments:
        # if not exp.has_already_run():
        exp.run()

    for exp in experiments:
        make_individual_exp_plots(exp)
