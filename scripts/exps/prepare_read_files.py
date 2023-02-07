"""Creates read files for the problems in exps."""

from solver_comparison.problem import problem_settings
from solver_comparison.problem.model import SOFTMAX
from solver_comparison.problem.problem import Problem

if __name__ == "__main__":
    for name, settings in problem_settings.items():
        (filename, K, N, L, alpha) = settings
        problem = Problem(
            model_type=SOFTMAX,
            filename=filename,
            K=K,
            N=N,
            L=L,
            alpha=alpha,
            beta=1.0,
        )
        print(name, problem)
        problem.load_model()
