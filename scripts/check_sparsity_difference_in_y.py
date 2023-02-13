import os

import numpy as np
import scipy as sp
from kmerexpr import simulate_reads as sr
from kmerexpr import transcriptome_reader as tr
from kmerexpr.rna_seq_reader import reads_to_y
from kmerexpr.utils import Problem
from scipy.sparse import coo_matrix

K = 14
problem = Problem(filename="sampled_genome_0.01.fsa", K=K, N=10**5, L=14)
ISO_FILE, READS_FILE, X_FILE, Y_FILE = problem.get_path_names()
READS_FILE = sr.simulate_reads(problem)
reads_to_y(problem.K, READS_FILE, Y_FILE=Y_FILE)
tr.transcriptome_to_x(problem.K, ISO_FILE, X_FILE, L=problem.L)


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


Y_FILE_SPARSE = Y_FILE
Y_FILE = Y_FILE.replace(".npz", ".npy")

y = np.load(Y_FILE)
nnz = np.sum(y != 0)

sp.sparse.save_npz(Y_FILE_SPARSE, sp.sparse.coo_matrix(y))
y_rec = sp.sparse.load_npz(Y_FILE_SPARSE).toarray().squeeze()

print(f"File size: {sizeof_fmt(os.path.getsize(Y_FILE))}")
print(f"Size of y: {len(y)} (4^{K})")
print(f"NNZ of y: {nnz} ({(nnz/len(y))*100:.2f}% dense)")
print(f"File size (sparse): {sizeof_fmt(os.path.getsize(Y_FILE_SPARSE))}")
print(f"same result: {np.all(y==y_rec)}")
