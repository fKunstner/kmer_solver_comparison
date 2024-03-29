from functools import partial

from solver_comparison.experiment import Experiment
from solver_comparison.problem.model import SIMPLEX, SOFTMAX
from solver_comparison.problem.problem import Problem
from solver_comparison.solvers.expgrad import ExpGrad
from solver_comparison.solvers.frank_wolfe import FrankWolfe
from solver_comparison.solvers.initializer import InitUniform
from solver_comparison.solvers.lbfgs import LBFGS
from solver_comparison.solvers.mg import MG
from solver_comparison.solvers.relsmooth import RelSmooth, RelSmoothLS


def all_optims_for(filename, K, N, L, alpha, max_iter):
    SoftmaxProblem = partial(Problem, model_type=SOFTMAX)
    SimplexProblem = partial(Problem, model_type=SIMPLEX)

    experiments = [
        Experiment(
            prob=SoftmaxProblem(
                filename=filename, K=K, N=N, L=L, alpha=alpha, beta=0.0
            ),
            opt=LBFGS(max_iter=max_iter),
            init=InitUniform(),
        ),
    ] + [
        Experiment(
            prob=SimplexProblem(
                filename=filename, K=K, N=N, L=L, alpha=alpha, beta=1.0
            ),
            opt=opt,
            init=InitUniform(),
        )
        for opt in [
            ExpGrad(max_iter=max_iter),
            FrankWolfe(max_iter=max_iter),
            MG(max_iter=max_iter),
            # RelSmooth(max_iter=max_iter),
            # RelSmoothLS(max_iter=max_iter),
        ]
    ]
    return list(experiments)


def make_experiment_for_opts(filename, K, N, L, alpha, softmax_opts, simplex_opts):
    SoftmaxProblem = partial(Problem, model_type=SOFTMAX)
    SimplexProblem = partial(Problem, model_type=SIMPLEX)
    return [
        Experiment(
            prob=SoftmaxProblem(
                filename=filename, K=K, N=N, L=L, alpha=alpha, beta=0.0
            ),
            opt=opt,
            init=InitUniform(),
        )
        for opt in softmax_opts
    ] + [
        Experiment(
            prob=SimplexProblem(
                filename=filename, K=K, N=N, L=L, alpha=alpha, beta=1.0
            ),
            opt=opt,
            init=InitUniform(),
        )
        for opt in simplex_opts
    ]
