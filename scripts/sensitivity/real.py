"""Model sensitivity experiment.

STEP 1: Plot test metrics as a function of
- number of reads, from 10^5 to 10^8
- length of reads, 100, 150, 200
- k-mer size, 8, 11, 14
- alpha parameter of Dirichlet that generate psi true, 0.01, 0.1, 1


Other things we need to check
- Seeds to obtain a measure of uncertainty?
- Optimization performance?
"""
from solver_comparison.experiment import Experiment
from solver_comparison.plotting import make_sensitivity_plot
from solver_comparison.problem.model import SIMPLEX
from solver_comparison.problem.problem import Problem
from solver_comparison.progressbar import progressbar
from solver_comparison.solvers.initializer import InitUniform
from solver_comparison.solvers.mg import MG

filename = "GRCh38_latest_rna.fna"
Ks = [2, 5, 8, 11, 14]
Ls = [100, 150, 200]
Ns = [10**3, 10**4, 10**5, 10**6, 10**7]
alphas = [0.01, 0.1, 1.0]
max_iters = [100]

experiments = [
    Experiment(
        prob=Problem(
            model_type=SIMPLEX,
            filename=filename,
            K=K,
            N=N,
            L=L,
            alpha=alpha,
            beta=1.0,
        ),
        opt=MG(max_iter=max_iter, tol=10**-6),
        init=InitUniform(),
    )
    for K in Ks
    for N in Ns
    for L in Ls
    for alpha in alphas
    for max_iter in max_iters
]

if __name__ == "__main__":

    import random

    random.seed(0)
    random.shuffle(experiments)

    for exp in progressbar(experiments):
        print(exp.as_dict())
        if exp.has_already_run():
            print("stored at ", exp.hash())
        else:
            exp.run()

    make_sensitivity_plot(
        experiments, Ks=Ks, Ns=Ns, Ls=Ls, alphas=alphas, max_iters=max_iters
    )
