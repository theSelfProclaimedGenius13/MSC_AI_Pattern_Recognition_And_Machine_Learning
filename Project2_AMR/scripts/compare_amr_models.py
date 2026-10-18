from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
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

splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.25, random_state=42
)
train_idx, test_idx = next(
    splitter.split(df, y, groups=groups)
)

def evaluate(name, columns):
    preprocessor = ColumnTransformer([
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical,
        ),
        ("numeric", "passthrough",
         [c for c in columns if c not in categorical]),
    ])

    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    model.fit(df.iloc[train_idx][columns], y.iloc[train_idx])
    pred = model.predict(df.iloc[test_idx][columns])
    actual = y.iloc[test_idx]

    return {
        "Model": name,
        "Accuracy": round(accuracy_score(actual, pred), 3),
        "Balanced accuracy": round(
            balanced_accuracy_score(actual, pred), 3
        ),
        "Macro F1": round(
            f1_score(actual, pred, average="macro"), 3
        ),
    }

results = [
    evaluate("Metadata only", categorical),
    evaluate("Metadata + DNA 4-mers", categorical + kmers),
]

result_df = pd.DataFrame(results)
print(result_df.to_string(index=False))
result_df.to_csv(RESULTS / "model_comparison.csv", index=False)
print("\nSaved comparison to project2_data/model_comparison.csv")


