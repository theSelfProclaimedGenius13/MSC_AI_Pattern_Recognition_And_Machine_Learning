from pathlib import Path
from collections import Counter
import itertools
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "project2_data"
GENOME_DIR = ROOT / "genomes" / "genomes_subset_extracted" / "ncbi_dataset" / "data"
OUTPUT_DIR = ROOT / "processed"
# All 256 possible DNA 4-mers
BASES = "ACGT"
KMERS = [
    "".join(x)
    for x in itertools.product(BASES, repeat=4)
]
KMER_INDEX = {kmer: i for i, kmer in enumerate(KMERS)}
K = 4


def read_fasta(path):
    """Yield DNA sequence lines from a FASTA file."""
    with path.open("rt", encoding="ascii", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith(">"):
                yield line.upper()


def extract_features(genome_dir):
    counts = np.zeros(len(KMERS), dtype=np.float64)
    valid_windows = 0

    # NCBI genome FASTA files can contain multiple contigs.
    # Do not count k-mers across contig boundaries.
    fasta_files = list(genome_dir.glob("*.fna"))
    if not fasta_files:
        fasta_files = (
            list(genome_dir.glob("*.fasta"))
            + list(genome_dir.glob("*.fa"))
        )

    for fasta in fasta_files:
        for sequence in read_fasta(fasta):
            if len(sequence) < K:
                continue

            for i in range(len(sequence) - K + 1):
                kmer = sequence[i:i + K]
                idx = KMER_INDEX.get(kmer)
                if idx is not None:
                    counts[idx] += 1
                    valid_windows += 1

    if valid_windows:
        counts /= valid_windows

    return counts, valid_windows


rows = []
genome_dirs = sorted(p for p in GENOME_DIR.iterdir() if p.is_dir())

for number, genome_dir in enumerate(genome_dirs, start=1):
    accession = genome_dir.name
    features, valid_windows = extract_features(genome_dir)

    if valid_windows == 0:
        print(f"WARNING: No valid 4-mers found for {accession}")
        continue

    row = {"assembly_accession": accession}
    row.update({kmer: features[i] for i, kmer in enumerate(KMERS)})
    row["valid_4mer_windows"] = valid_windows
    rows.append(row)

    if number % 50 == 0 or number == len(genome_dirs):
        print(f"Processed {number}/{len(genome_dirs)} genomes")

feature_df = pd.DataFrame(rows)
output_path = OUTPUT_DIR / "genome_4mer_features.csv"
feature_df.to_csv(output_path, index=False)

print("\nFeature extraction complete.")
print("Genomes processed:", len(feature_df))
print("Number of 4-mer features:", len(KMERS))
print("Feature table shape:", feature_df.shape)
print("Saved to:", output_path)



