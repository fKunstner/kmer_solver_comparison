from functools import partial

from solver_comparison.experiment import Experiment
from solver_comparison.plotting import make_comparison_plots, make_individual_exp_plots
from solver_comparison.problem.model import SIMPLEX, SOFTMAX
from solver_comparison.problem.problem import Problem
from solver_comparison.solvers.expgrad import ExpGrad
from solver_comparison.solvers.frank_wolfe import AwayFrankWolfe, FrankWolfe
from solver_comparison.solvers.initializer import Initializer, InitUniform
from solver_comparison.solvers.lbfgs import LBFGS

problem = "small"

if problem == "small":
    filename = "test5.fsa"
    K, N, L, alpha = 8, 100, 14, 0.1
    max_iter = 100
elif problem == "medium":
    filename = "sampled_genome_0.01.fsa"
    max_iter = 200
    K, N, L, alpha = 14, 500000, 50, 0.1
else:
    raise ValueError(f"Problem {problem} unknown")

SoftmaxProblem = partial(Problem, model_type=SOFTMAX)
SimplexProblem = partial(Problem, model_type=SIMPLEX)


experiments = [
    Experiment(
        prob=SoftmaxProblem(filename=filename, K=K, N=N, L=L, alpha=alpha, beta=1.0),
        opt=LBFGS(max_iter=max_iter),
        init=InitUniform(),
    ),
] + [
    Experiment(
        prob=SimplexProblem(filename=filename, K=K, N=N, L=L, alpha=alpha, beta=1.0),
        opt=opt,
        init=InitUniform(),
    )
    for opt in [
        ExpGrad(max_iter=max_iter),
        FrankWolfe(max_iter=max_iter),
        AwayFrankWolfe(max_iter=max_iter),
    ]
]


if __name__ == "__main__":
    for exp in experiments:
        print(exp.as_dict())
        print(exp.as_str())
        if not exp.has_already_run():
            exp.run()

    for exp in experiments:
        make_individual_exp_plots(exp)

    make_comparison_plots(experiments)
