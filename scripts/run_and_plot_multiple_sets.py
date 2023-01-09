from tqdm import tqdm

from solver_comparison.exp_defs import all_optims_for
from solver_comparison.plotting import make_all_plots

if __name__ == "__main__":
    Ns = [10**i for i in [0, 1, 2, 3, 4]]
    alphas = [1.0, 0.1, 0.01, 0.001]

    experiments_sets = [
        all_optims_for(
            filename="test5.fsa",
            K=14,
            N=N,
            L=14,
            alpha=alpha,
            max_iter=1000,
        )
        for N in Ns
        for alpha in alphas
    ] + [
        all_optims_for(
            filename="sampled_genome_0.01.fsa",
            K=14,
            N=N,
            L=50,
            alpha=alpha,
            max_iter=10000,
        )
        for N in Ns
        for alpha in alphas
    ]

    for experiments_set in tqdm(experiments_sets):
        for exp in tqdm(experiments_set):
            print(exp.as_dict())
            if not exp.has_already_run():
                exp.run()
            print("stored at ", exp.hash())

    for experiments_set in tqdm(experiments_sets[23:]):
        make_all_plots(experiments_set)
