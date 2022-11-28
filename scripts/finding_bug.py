import numpy as np
import simulate_reads as sr
import transcriptome_reader as tr
from multinomial_simplex_model import multinomial_simplex_model
from rna_seq_reader import reads_to_y
from utils import Model_Parameters
from utils import Problem as KmerExprProblem
from utils import load_lengths

if __name__ == "__main__":

    # filename = "test5.fsa"
    filename = "sampled_genome_0.01.fsa"
    K = 8
    N = 1000
    L = 14
    alpha = 0.1
    beta = 1.0

    sampled_genome_dataset = {
        "filename": filename,
        "K": K,
        "N": N,
        "L": L,
        "alpha": alpha,
    }

    kmer_problem = KmerExprProblem(**sampled_genome_dataset)

    (ISO_FILE, READS_FILE, X_FILE, Y_FILE) = kmer_problem.get_path_names()
    sr.simulate_reads(kmer_problem, force_repeat=False)
    reads_to_y(K, READS_FILE, Y_FILE=Y_FILE)
    tr.transcriptome_to_x(K, ISO_FILE, X_FILE, L=L)
    lengths = load_lengths(filename, N, L)

    model_params = Model_Parameters(
        model_type="simplex",
        solver_name="exp_grad",
        beta=1.0,
        lrs="warmstart",
        init_iterates="uniform",
    )

    model = multinomial_simplex_model(
        X_FILE, Y_FILE, beta=beta, lengths=lengths, solver_name="exp_grad"
    )
    # model = model_params.initialize_model(X_FILE, Y_FILE, lengths=lengths)

    model.fit(model_params, n_iters=50)
