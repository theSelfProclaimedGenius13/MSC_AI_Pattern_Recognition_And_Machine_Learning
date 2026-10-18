from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "project2_data"
GENOME_DIR = DATA / "genomes" / "genomes_subset_extracted" / "ncbi_dataset" / "data"

phenotypes = pd.read_csv(DATA / "intermediate" / "phenotypes_subset.csv", dtype=str)
assemblies = set(
    p.name
    for p in GENOME_DIR.iterdir()
    if p.is_dir()
)

phenotypes["assembly_accession"] = (
    phenotypes["assembly_accession"].astype(str).str.strip()
)

phenotype_accessions = set(phenotypes["assembly_accession"].dropna())
matched = phenotype_accessions & assemblies
missing = phenotype_accessions - assemblies

print("Phenotype records:", len(phenotypes))
print("Unique phenotype accessions:", len(phenotype_accessions))
print("Genome directories found:", len(assemblies))
print("Matching accessions:", len(matched))
print("Missing accessions:", len(missing))

print("\nLabels:")
print(phenotypes["label"].value_counts(dropna=False))

print("\nOrganism-antibiotic-label counts:")
print(
    phenotypes.groupby(
        ["organism_group", "antibiotic", "label"],
        dropna=False
    ).size().to_string()
)


