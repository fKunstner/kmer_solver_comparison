from solver_comparison.exp_defs import all_optims_for

experiments = all_optims_for(
    filename="sampled_genome_0.01.fsa",
    K=14,
    N=500000,
    L=50,
    alpha=0.1,
    max_iter=200,
)
