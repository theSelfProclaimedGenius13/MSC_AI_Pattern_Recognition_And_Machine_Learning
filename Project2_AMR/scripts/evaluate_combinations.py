from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "project2_data" / "processed"
RESULTS = ROOT / "project2_data" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(DATA / "model_dataset.csv")

# Select only the 256 DNA 4-mer frequency features.
kmers = [
    c for c in df.columns
    if len(c) == 4 and set(c).issubset(set("ACGT"))
]

combinations = df[["organism_group", "antibiotic"]].drop_duplicates()
results = []

for _, combo in combinations.iterrows():
    organism = combo["organism_group"]
    antibiotic = combo["antibiotic"]

    subset = df[
        (df["organism_group"] == organism)
        & (df["antibiotic"] == antibiotic)
    ].copy()

    subset = subset.dropna(subset=kmers + ["label", "assembly_accession"])

    X = subset[kmers]
    y = subset["label"].map({"susceptible": 0, "resistant": 1})
    groups = subset["assembly_accession"]

    if y.nunique() < 2 or groups.nunique() < 4:
        print(f"Skipping {organism} / {antibiotic}: insufficient data")
        continue

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42,
    )
    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    predictions = model.predict(X.iloc[test_idx])
    actual = y.iloc[test_idx]

    results.append({
        "Organism": organism,
        "Antibiotic": antibiotic,
        "Records": len(subset),
        "Test records": len(test_idx),
        "Test genomes": groups.iloc[test_idx].nunique(),
        "Accuracy": accuracy_score(actual, predictions),
        "Balanced accuracy": balanced_accuracy_score(actual, predictions),
        "Macro F1": f1_score(
            actual, predictions, average="macro", zero_division=0
        ),
    })

result_df = pd.DataFrame(results)

if not result_df.empty:
    metric_columns = ["Accuracy", "Balanced accuracy", "Macro F1"]
    result_df[metric_columns] = result_df[metric_columns].round(3)

    print("\n=== PERFORMANCE BY ORGANISM–ANTIBIOTIC COMBINATION ===")
    print(result_df.to_string(index=False))

    result_df.to_csv(RESULTS / "combination_results.csv", index=False)
    print("\nSaved to:", RESULTS / "combination_results.csv")
else:
    print("No combinations could be evaluated.")


