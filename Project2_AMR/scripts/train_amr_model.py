from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "project2_data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(DATA / "model_dataset.csv")

# The 256 4-mer frequency columns.
kmer_columns = [
    col for col in df.columns
    if len(col) == 4 and set(col).issubset(set("ACGT"))
]

categorical_columns = ["organism_group", "antibiotic"]
feature_columns = kmer_columns + categorical_columns

X = df[feature_columns].copy()
y = df["label"].map({
    "susceptible": 0,
    "resistant": 1,
})
groups = df["assembly_accession"]

if y.isna().any():
    raise ValueError("Unexpected or missing phenotype labels.")

print("Records:", len(df))
print("DNA features:", len(kmer_columns))
print("Categorical features:", categorical_columns)
print("Unique genomes:", groups.nunique())

# Split by genome, not by individual phenotype row.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

train_genomes = set(groups.iloc[train_idx])
test_genomes = set(groups.iloc[test_idx])

assert train_genomes.isdisjoint(test_genomes)

print("\nTraining records:", len(train_idx))
print("Testing records:", len(test_idx))
print("Training genomes:", len(train_genomes))
print("Testing genomes:", len(test_genomes))
print("Train/test genome overlap:", len(train_genomes & test_genomes))

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns,
        ),
        ("dna", "passthrough", kmer_columns),
    ]
)

model = Pipeline([
    ("preprocess", preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    ),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\n=== TEST RESULTS ===")
print(f"Accuracy:          {accuracy_score(y_test, predictions):.3f}")
print(f"Balanced accuracy: {balanced_accuracy_score(y_test, predictions):.3f}")
print(f"Macro F1:          {f1_score(y_test, predictions, average='macro'):.3f}")

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Susceptible", "Resistant"],
    zero_division=0,
))

print("Confusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

output_path = MODEL_DIR / "random_forest_amr_model.joblib"

import joblib
joblib.dump(model, output_path)

print("\nSaved trained model to:", output_path)


