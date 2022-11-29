# Solver comparison

Benchmarking optimizers for 
[`kmerexpr`](https://github.com/bob-carpenter/kmers).

## Installation 

1. The library assumes `kmerexpr` is installed. 
    If this is not yet the case, either 
    - clone the repository and install it in editable mode with `-e` if working on `kmerexpr`)
        ```
        git clone https://github.com/bob-carpenter/kmers
        cd kmers 
        pip install -e . 
        ```
    - Install it directly from git 
        ```
        pip install git+https://github.com/bob-carpenter/kmers
        ```
2. To install this library and have access to the scripts, 
    ```
    git clone https://github.com/fKunstner/kmer_solver_comparison
    cd kmer_solver_comparison
    pip install -e . 
    ```

### Configuration 
By default, `solver_comparison` saves data in `~/solver_comparison`.

This behavior can be modified by setting the environment variable
`KMEREXPR_BENCH_DATA_ROOT`.
- On OSX/Unix, `source env.sh` with `env.sh` containing
  ```
  export KMEREXPR_BENCH_DATA_ROOT=~/path/to/data/root
  ```
- On Windows, `call env.bat` with `env.bat` containing
  ```
  set KMEREXPR_BENCH_DATA_ROOT=C:\User\user\path\to\data\root
  ```


## Usage



## Extending the library

Main files/concepts: 

**Problem definitions**
- [`Model`](src/solver_comparison/problem/model.py) 
  Definition of the loss function and gradient given a dataset.  
  Wraps the code in
  [`multinomial_model`](../multinomial_model.py),
  [`multinomial_simplex_model`](../multinomial_simplex_model.py) 
  and [`normal_model`](../normal_model.py).
- [`Problem`](src/solver_comparison/problem/problem.py): Combination of a `Model` and a dataset.   
  Wraps the code in
  [`simulate_reads`](../simulate_reads.py) 
  [`transcriptome_reader`](../transcriptome_reader.py) 
  [`rna_seq_reader`](../rna_seq_reader.py) 

**Solvers**
- [`Initializer`](src/solver_comparison/solvers/initializer.py): Initialization strategies
- [`Optimizer`](src/solver_comparison/solvers/optimizer.py): Provide a generic interface to different optimizers

**Benchmarking/Running**
- [`Experiment`](src/solver_comparison/experiment.py): Wraps a `Problem`, `Initializer`, `Optimizer`.  
  Runs an experiment and logs the results with some help 

- **Data logging**
- [`GlobalLogger.DataLogger`](src/solver_comparison/logging/datalogger.py): 
  Saving arbitrary data to a `.csv` depending on the experiment hash. 

**Missing** 
- Moving other optimizers and problems in this format (currently only plain GD)
- Managing experiment data when finished running (aka making plots)

#### Coding style 

This code is formatted with 
[Black](https://github.com/psf/black) and 
[isort](https://github.com/PyCQA/isort).
Docstrings in 
[Google style](https://google.github.io/styleguide/pyguide.html#s3.8.1-comments-in-doc-strings)
are checked with
[docformatter](https://github.com/PyCQA/docformatter).

