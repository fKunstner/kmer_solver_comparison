import os.path
from dataclasses import dataclass, field

from kmerexpr import simulate_reads as sr
from kmerexpr import transcriptome_reader as tr
from kmerexpr.rna_seq_reader import reads_to_y
from kmerexpr.utils import Problem as KmerExprProblem
from kmerexpr.utils import load_lengths, load_simulation_parameters

from solver_comparison.problem.model import Model


@dataclass
class Problem:
    """Wrapper around the datasets and model in kmerexpr.

    Args:
        model_type: Parametrization to use
            for the model.
        filename: Path to the .fsa file containing the data
        N: Number of reads
        K: Number of Kmers
        L: Length of reads
        alpha: Strength of Dirichlet prior for simulating reads
        beta: Regularization parameter of the model
    """

    model_type: str
    filename: str
    N: int
    K: int
    L: int
    alpha: float
    beta: float
    kmer_problem: KmerExprProblem = field(init=False)

    def __post_init__(self):
        self.kmer_problem = KmerExprProblem(
            self.filename, K=self.K, N=self.N, L=self.L, alpha=self.alpha
        )

    def load_model(self) -> Model:
        """Creates data for the problem and loads the model -- Time hungry."""
        (ISO_FILE, READS_FILE, X_FILE, Y_FILE) = self.kmer_problem.get_path_names()
        sr.simulate_reads(self.kmer_problem, force_repeat=False)
        if not os.path.exists(Y_FILE):
            reads_to_y(self.K, READS_FILE, Y_FILE=Y_FILE)
        if not os.path.exists(X_FILE):
            tr.transcriptome_to_x(self.K, ISO_FILE, X_FILE, L=self.L)
        lengths = load_lengths(self.filename, self.N, self.L)

        return Model(self.model_type, X_FILE, Y_FILE, beta=self.beta, lengths=lengths)

    def load_simulation_parameters(self):
        return load_simulation_parameters(self.kmer_problem)
