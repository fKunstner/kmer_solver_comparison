from functools import partial

from solver_comparison.experiment import Experiment
from solver_comparison.problem.model import SIMPLEX, SOFTMAX
from solver_comparison.problem.problem import Problem
from solver_comparison.solvers.expgrad import ExpGrad
from solver_comparison.solvers.frank_wolfe import FrankWolfe
from solver_comparison.solvers.initializer import InitUniform
from solver_comparison.solvers.lbfgs import LBFGS
from solver_comparison.solvers.mg import MG

filename = "sampled_genome_0.01.fsa"
K, N, L, alpha = 14, 500000, 50, 0.1
max_iter = 100

SoftmaxProblem = partial(Problem, model_type=SOFTMAX)
SimplexProblem = partial(Problem, model_type=SIMPLEX)

experiments = [
    Experiment(
        prob=SoftmaxProblem(filename=filename, K=K, N=N, L=L, alpha=alpha, beta=0.0),
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
        MG(max_iter=max_iter),
    ]
]
