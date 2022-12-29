import gzip
import os
import shutil
import urllib.request
from pathlib import Path

from kmerexpr import sample_genome_data

from solver_comparison.kmerexpr_data import kmerexpr_data_path

test_name = "test5.fsa"
test_url = "https://raw.githubusercontent.com/bob-carpenter/kmers/be5d806b928253cbc94d58e59fa2378d79c97d00/data/test5.fsa"


big_zip_name = "GRCh38_latest_rna.fna.gz"
big_url = "https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/annotation/GRCh38_latest/refseq_identifiers/GRCh38_latest_rna.fna.gz"


def download_if_not_exists(url, filename):
    filepath = os.path.join(kmerexpr_data_path(), filename)
    if not os.path.exists(filepath):
        print("Beginning download of ", url)
        urllib.request.urlretrieve(url, filepath)


if __name__ == "__main__":
    Path(kmerexpr_data_path()).mkdir(parents=True, exist_ok=True)

    print(f"Saving data in {kmerexpr_data_path()}")

    download_if_not_exists(test_url, test_name)

    download_if_not_exists(big_url, big_zip_name)

    filepath = os.path.join(kmerexpr_data_path(), big_zip_name)
    extracted_filepath = filepath.replace(".gz", "")
    if not os.path.exists(extracted_filepath):
        print("Unpacking file ", filepath)
        with gzip.open(filepath, "rb") as f_in:
            with open(extracted_filepath, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    for p in [0.001, 0.01, 0.1]:
        sampled_file = "sampled_genome" + "_" + str(p) + ".fsa"
        if not os.path.exists(os.path.join(kmerexpr_data_path(), sampled_file)):
            print(f"Subsampling file to p={p}")
            sample_genome_data.sample_genome_data(extracted_filepath, sampled_file, p)
            shutil.move(sampled_file, os.path.join(kmerexpr_data_path(), sampled_file))
