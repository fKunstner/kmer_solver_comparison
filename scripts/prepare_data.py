import gzip
import os
import shutil
import urllib.request
from pathlib import Path

import sample_genome_data
from solver_comparison.kmerexpr import kmerexpr_data_path

fsa_urls = {
    "test5.fsa": "https://raw.githubusercontent.com/bob-carpenter/kmers/be5d806b928253cbc94d58e59fa2378d79c97d00/data/test5.fsa"
}
fna_gz_urls = {
    "GRCh38_latest_rna.fna.gz": "https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/annotation/GRCh38_latest/refseq_identifiers/GRCh38_latest_rna.fna.gz"
}


if __name__ == "__main__":
    kmers_data_path = kmerexpr_data_path()
    Path(kmers_data_path).mkdir(parents=True, exist_ok=True)

    print(f"Saving data in {kmers_data_path}")
    for filename, url in fsa_urls.items():
        print("Beginning download of ", url)
        filepath = os.path.join(kmers_data_path, filename)
        urllib.request.urlretrieve(url, filepath)

    for filename, url in fna_gz_urls.items():
        print("Beginning download of ", url)
        filepath = os.path.join(kmers_data_path, filename)
        urllib.request.urlretrieve(url, filepath)

        print("Unpacking file ", filepath)
        extracted_filepath = filepath.replace(".gz", "")
        with gzip.open(filepath, "rb") as f_in:
            with open(extracted_filepath, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        for p in [0.001, 0.01, 0.1]:
            print(f"Subsampling file to p={p}")
            sampled_file = "sampled_genome" + "_" + str(p) + ".fsa"
            sample_genome_data.sample_genome_data(extracted_filepath, sampled_file, p)
            shutil.move(sampled_file, os.path.join(kmers_data_path, sampled_file))
