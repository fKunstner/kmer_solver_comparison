"""List the `.fsa` files available in kmerexpr to make test problems."""
import os

import solver_comparison.kmerexpr

if __name__ == "__main__":
    for file in os.listdir(solver_comparison.kmerexpr.kmerexpr_data_path()):
        if file.endswith(".fsa"):
            print(file)
