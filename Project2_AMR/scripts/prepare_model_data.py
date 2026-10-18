from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "project2_data"
INTERMEDIATE = ROOT / "intermediate"
PROCESSED = ROOT / "processed"

phenotypes = pd.read_csv(
    INTERMEDIATE / "phenotypes_subset.csv",
    dtype={"assembly_accession": str},
)

features = pd.read_csv(
    PROCESSED / "genome_4mer_features.csv",
    dtype={"assembly_accession": str},
)

group_cols = ["assembly_accession", "antibiotic"]

conflicts = (
    phenotypes.groupby(group_cols)["label"]
    .nunique()
    .reset_index(name="number_of_labels")
)

conflicts = conflicts[conflicts["number_of_labels"] > 1]

print("Phenotype rows:", len(phenotypes))
print("Unique genomes in phenotypes:",
      phenotypes["assembly_accession"].nunique())
print("Feature rows:", len(features))
print("Conflicting genome-antibiotic pairs:", len(conflicts))

if not conflicts.empty:
    print("\nExamples of conflicting pairs:")
    print(conflicts.head(10).to_string(index=False))
    raise ValueError(
        "Resolve conflicting phenotype labels before modelling."
    )

model_data = phenotypes.merge(
    features,
    on="assembly_accession",
    how="left",
    validate="many_to_one",
    indicator=True,
)

print("\nMerge status:")
print(model_data["_merge"].value_counts())

model_data = model_data.drop(columns="_merge")

if model_data["ACGT"].isna().any():
    raise ValueError("Some phenotype records have no genome features.")

output = PROCESSED / "model_dataset.csv"
model_data.to_csv(output, index=False)

print("\nModel dataset shape:", model_data.shape)
print("Saved to:", output)


