from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "project2_data" / "processed"
RESULTS = ROOT / "project2_data" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(DATA / "model_dataset.csv")

categorical = ["organism_group", "antibiotic"]
kmers = [
    c for c in df.columns
    if len(c) == 4 and set(c).issubset(set("ACGT"))
]

y = df["label"].map({"susceptible": 0, "resistant": 1})
groups = df["assembly_accession"]

if y.isna().any() or groups.isna().any():
    raise ValueError("Missing labels or genome accessions found.")

# Three feature configurations for a fair comparison.
experiments = {
    "Metadata only": categorical,
    "DNA only": kmers,
    "DNA + metadata": categorical + kmers,
}

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

fold_results = []

for experiment_name, columns in experiments.items():
    print(f"\nRunning: {experiment_name}")

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(df[columns], y, groups=groups), start=1
    ):
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])

        if not train_groups.isdisjoint(test_groups):
            raise RuntimeError("Genome leakage detected!")

        transformers = []

        categorical_used = [
            col for col in categorical if col in columns
        ]
        numeric_used = [
            col for col in columns if col not in categorical
        ]

        if categorical_used:
            transformers.append((
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_used,
            ))

        if numeric_used:
            transformers.append((
                "dna",
                "passthrough",
                numeric_used,
            ))

        preprocessing = ColumnTransformer(transformers)

        model = Pipeline([
            ("preprocess", preprocessing),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ])

        X_train = df.iloc[train_idx][columns]
        X_test = df.iloc[test_idx][columns]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        bal_acc = balanced_accuracy_score(y_test, predictions)
        macro_f1 = f1_score(
            y_test, predictions, average="macro", zero_division=0
        )

        fold_results.append({
            "Model": experiment_name,
            "Fold": fold,
            "Train records": len(train_idx),
            "Test records": len(test_idx),
            "Test genomes": len(test_groups),
            "Balanced accuracy": bal_acc,
            "Macro F1": macro_f1,
        })

        print(
            f"  Fold {fold}/5: "
            f"balanced accuracy={bal_acc:.3f}, "
            f"macro-F1={macro_f1:.3f}, "
            f"test records={len(test_idx)}"
        )

results = pd.DataFrame(fold_results)
results.to_csv(RESULTS / "cross_validation_folds.csv", index=False)

summary = (
    results.groupby("Model")
    .agg(
        Mean_balanced_accuracy=("Balanced accuracy", "mean"),
        SD_balanced_accuracy=("Balanced accuracy", "std"),
        Mean_macro_F1=("Macro F1", "mean"),
        SD_macro_F1=("Macro F1", "std"),
    )
    .reset_index()
)

for col in summary.columns:
    if col != "Model":
        summary[col] = summary[col].round(3)

print("\n=== FIVE-FOLD GROUPED CROSS-VALIDATION SUMMARY ===")
print(summary.to_string(index=False))

summary.to_csv(RESULTS / "cross_validation_summary.csv", index=False)
print("\nSaved fold results to:", RESULTS / "cross_validation_folds.csv")
print("Saved summary to:", RESULTS / "cross_validation_summary.csv")


