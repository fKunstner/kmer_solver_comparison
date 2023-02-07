from typing import Tuple

LARGE_FILE = "GRCh38_latest_rna.fna"
MEDIUM_FILE = "sampled_genome_0.1.fsa"
SMALL_FILE = "sampled_genome_0.01.fsa"
TEST_FILE = "test5.fsa"

A_SPARSE = 0.01
A_MEDIUM = 0.1
A_DENSE = 0.5


def settings(filename, K, N, L, alpha) -> Tuple[str, int, int, int, float]:
    return filename, K, N, L, alpha


problem_settings = {
    "debug-sparse": settings(TEST_FILE, K=8, N=100, L=14, alpha=A_SPARSE),
    "debug-medium": settings(TEST_FILE, K=8, N=100, L=14, alpha=A_MEDIUM),
    "debug-dense": settings(TEST_FILE, K=8, N=100, L=14, alpha=A_DENSE),
    "small-sparse": settings(SMALL_FILE, K=14, N=10**6, L=50, alpha=A_SPARSE),
    "small-medium": settings(SMALL_FILE, K=14, N=10**6, L=50, alpha=A_MEDIUM),
    "small-dense": settings(SMALL_FILE, K=14, N=10**6, L=50, alpha=A_DENSE),
    "medium-sparse": settings(MEDIUM_FILE, K=14, N=10**7, L=100, alpha=A_SPARSE),
    "medium-medium": settings(MEDIUM_FILE, K=14, N=10**7, L=100, alpha=A_MEDIUM),
    "medium-dense": settings(MEDIUM_FILE, K=14, N=10**7, L=100, alpha=A_DENSE),
    "large-sparse": settings(LARGE_FILE, K=14, N=10**8, L=200, alpha=A_SPARSE),
    "large-medium": settings(LARGE_FILE, K=14, N=10**8, L=200, alpha=A_MEDIUM),
    "large-dense": settings(LARGE_FILE, K=14, N=10**8, L=200, alpha=A_DENSE),
    "massive-sparse": settings(LARGE_FILE, K=14, N=10**9, L=200, alpha=A_SPARSE),
    "massive-medium": settings(LARGE_FILE, K=14, N=10**9, L=200, alpha=A_MEDIUM),
    "massive-dense": settings(LARGE_FILE, K=14, N=10**9, L=200, alpha=A_DENSE),
}
