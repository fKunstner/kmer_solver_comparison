from pathlib import Path


def kmerexpr_data_path():
    from kmerexpr import sample_genome_data

    return Path(sample_genome_data.__file__).parents[1] / "data"
