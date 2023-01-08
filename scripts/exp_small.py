from solver_comparison.exp_defs import all_optims_for

experiments = all_optims_for(
    filename="test5.fsa", K=8, N=100, L=14, alpha=0.1, max_iter=50
)
