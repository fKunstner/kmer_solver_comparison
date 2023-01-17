"""List the `.fsa` files available in kmerexpr to make test problems."""
import os

from solver_comparison.kmerexpr_data import kmerexpr_data_path

if __name__ == "__main__":
    for file in os.listdir(kmerexpr_data_path()):
        if file.endswith(".fsa"):
            print(file)
