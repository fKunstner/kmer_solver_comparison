import numpy as np
from solver_comparison.experiment import Experiment
from solver_comparison.plotting import make_individual_exp_plots
from solver_comparison.problem.problem import Problem
from solver_comparison.solvers.initializer import Initializer
from solver_comparison.solvers.optimizer import LBFGS, ExpGrad

sampled_genome_dataset = {
    "filename": "sampled_genome_0.01.fsa",
    "K": 8,
    "N": 1_000,
    "L": 14,
    "alpha": 0.1,
    "beta": 1.0,
}

problems_logistic = [Problem(model_name="Logistic", **sampled_genome_dataset)]
problems_simplex = [Problem(model_name="Simplex", **sampled_genome_dataset)]

experiments_simplex = [
    Experiment(
        prob=prob,
        opt=opt,
        init=Initializer("simplex_uniform"),
    )
    for prob in problems_simplex
    for opt in [ExpGrad()]
]

experiments_logistic = [
    Experiment(
        prob=prob,
        opt=opt,
        init=Initializer("zero"),
    )
    for prob in problems_logistic
    for opt in [LBFGS()]
]

experiments = experiments_simplex


if __name__ == "__main__":

    # np.seterr(all="raise")

    for exp in experiments:
        # if not exp.has_already_run():
        exp.run()
        # make_individual_exp_plots(exp)
