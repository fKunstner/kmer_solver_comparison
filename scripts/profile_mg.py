from solver_comparison.experiment import Experiment
from solver_comparison.problem import problem_settings
from solver_comparison.problem.model import SIMPLEX
from solver_comparison.problem.problem import Problem
from solver_comparison.solvers.initializer import InitUniform
from solver_comparison.solvers.mg import MG

max_iter = 100
filename, K, N, L, alpha = problem_settings["medium-sparse"]

exp = Experiment(
    prob=Problem(
        model_type=SIMPLEX,
        filename=filename,
        K=K,
        N=N,
        L=L,
        alpha=alpha,
        beta=1.0,
    ),
    opt=MG(max_iter=max_iter),
    init=InitUniform(),
)


if __name__ == "__main__":
    exp.run()
