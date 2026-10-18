import pandas as pd
from pathlib import Path

base = Path(__file__).resolve().parents[1] / "project2_data" / "intermediate"
source = base / "phenotypes_binary_SR.csv"

df = pd.read_csv(source, low_memory=False)
df = df.dropna(subset=["assembly_accession"]).copy()

targets = {
    ("Salmonella enterica", "ampicillin"),
    ("Salmonella enterica", "tetracycline"),
    ("E.coli and Shigella", "ampicillin"),
    ("E.coli and Shigella", "tetracycline"),
    ("Campylobacter jejuni", "ciprofloxacin"),
    ("Campylobacter jejuni", "tetracycline"),
}

df = df[
    df.apply(
        lambda r: (r["organism_group"], r["antibiotic"]) in targets,
        axis=1,
    )
].copy()

# Avoid counting the same assembly twice for the same label
# and organism-antibiotic combination.
df = df.drop_duplicates(
    subset=[
        "organism_group",
        "antibiotic",
        "assembly_accession",
        "label",
    ]
)

parts = []
for (organism, antibiotic), group in df.groupby(
    ["organism_group", "antibiotic"]
):
    for label in ["resistant", "susceptible"]:
        candidates = group[group["label"] == label]
        sampled = candidates.sample(
            n=min(100, len(candidates)),
            random_state=42,
        )
        parts.append(sampled)

subset = pd.concat(parts, ignore_index=True)
subset.to_csv(base / "phenotypes_subset.csv", index=False)

assemblies = sorted(subset["assembly_accession"].unique())
(base / "assemblies_subset.txt").write_text(
    "\n".join(assemblies) + "\n",
    encoding="utf-8",
)

print("Phenotype records:", len(subset))
print("Unique assemblies:", len(assemblies))
print("\nCounts by combination and label:")
print(
    subset.groupby(["organism_group", "antibiotic", "label"])
    .size()
    .to_string()
)
print("\nCreated:", base / "phenotypes_subset.csv")
print("Created:", base / "assemblies_subset.txt")


