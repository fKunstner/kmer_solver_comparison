from solver_comparison.exp_defs import all_optims_for

experiments = all_optims_for(
    filename="sampled_genome_0.1.fsa",
    K=14,
    N=5000000,
    L=100,
    alpha=0.1,
    max_iter=500,
)
