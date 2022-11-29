from dataclasses import dataclass, field

import simulate_reads as sr
import transcriptome_reader as tr
from rna_seq_reader import reads_to_y
from solver_comparison.problem.model import KmerModel, KmerModelName, model_classes
from solver_comparison.serialization import Serializable
from utils import Problem as KmerExprProblem
from utils import load_lengths, load_simulation_parameters


@dataclass
class Problem(Serializable):
    """Wrapper around the datasets and model in kmerexpr.

    Args:
        model_name (Literal["Logistic", "Simplex"]): Parametrization to use
            for the model.
        filename (str): Path to the .fsa file containing the data
        N (int): Number of reads
        K (int): Number of Kmers
        L (int): Length of reads
        alpha (float): Strength of Dirichlet prior for simulating reads
        beta (float): Regularization parameter of the model
    """

    model_name: KmerModelName
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

    def load_model(self) -> KmerModel:
        """Creates data for the problem and loads the model -- Time hungry."""
        (ISO_FILE, READS_FILE, X_FILE, Y_FILE) = self.kmer_problem.get_path_names()
        sr.simulate_reads(self.kmer_problem, force_repeat=False)
        reads_to_y(self.K, READS_FILE, Y_FILE=Y_FILE)
        tr.transcriptome_to_x(self.K, ISO_FILE, X_FILE, L=self.L)
        lengths = load_lengths(self.filename, self.N, self.L)

        return model_classes[self.model_name](
            X_FILE, Y_FILE, beta=self.beta, lengths=lengths, solver_name=None
        )

    def load_simulation_parameters(self):
        return load_simulation_parameters(self.kmer_problem)


test_problem_logistic = Problem(
    model_name="Logistic", filename="test5.fsa", K=8, N=1_000, L=14, alpha=0.1, beta=0.0
)
test_problem_simplex = Problem(
    model_name="Simplex", filename="test5.fsa", K=8, N=1_000, L=14, alpha=0.1, beta=0.0
)
