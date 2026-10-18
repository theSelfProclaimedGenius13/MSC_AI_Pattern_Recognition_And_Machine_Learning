from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    accuracy_score,
    ConfusionMatrixDisplay,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "project2_data"
CHART_DIR = DATA_DIR / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

# Load the dataset
df = pd.read_csv(DATA_DIR / "processed" / "model_dataset.csv")

# Identify DNA 4-mer columns
kmers = [
    col for col in df.columns
    if len(col) == 4 and set(col).issubset(set("ACGT"))
]

required = ["organism_group", "antibiotic", "assembly_accession", "label"]
missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n"
        f"Available columns: {df.columns.tolist()}"
    )

if not kmers:
    raise ValueError("No DNA 4-mer columns were found.")

X = df[["organism_group", "antibiotic"] + kmers]
y = df["label"].astype(str).str.strip().str.lower()
groups = df["assembly_accession"]

# Standardize the expected label names
label_map = {
    "s": "susceptible",
    "susceptible": "susceptible",
    "r": "resistant",
    "resistant": "resistant",
}
y = y.map(label_map)

if y.isna().any():
    raise ValueError(
        "Unexpected target labels: "
        f"{df.loc[y.isna(), 'label'].unique().tolist()}"
    )

class_labels = ["susceptible", "resistant"]

print(f"Records: {len(df)}")
print(f"Unique genomes: {groups.nunique()}")
print(f"DNA 4-mer features: {len(kmers)}")
print("\nClass counts:")
print(y.value_counts())

# Define the same model configuration used in the comparison
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            ["organism_group", "antibiotic"],
        ),
        ("dna", "passthrough", kmers),
    ]
)

model = Pipeline(
    steps=[
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
    ]
)

# Generate an out-of-fold prediction for every record
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

predictions = np.empty(len(df), dtype=object)
predictions[:] = None

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups), start=1
):
    train_genomes = set(groups.iloc[train_idx])
    test_genomes = set(groups.iloc[test_idx])

    overlap = train_genomes.intersection(test_genomes)
    if overlap:
        raise RuntimeError("Genome leakage detected between train and test.")

    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    predictions[test_idx] = model.predict(X.iloc[test_idx])

    fold_score = balanced_accuracy_score(
        y.iloc[test_idx], predictions[test_idx]
    )
    print(
        f"Fold {fold}/5: "
        f"test records={len(test_idx)}, "
        f"balanced accuracy={fold_score:.3f}"
    )

if any(pred is None for pred in predictions):
    raise RuntimeError("Some records did not receive an out-of-fold prediction.")

# Overall metrics from all held-out predictions
cm = confusion_matrix(y, predictions, labels=class_labels)

print("\n=== POOLED OUT-OF-FOLD RESULTS ===")
print(f"Accuracy: {accuracy_score(y, predictions):.3f}")
print(f"Balanced accuracy: {balanced_accuracy_score(y, predictions):.3f}")
print("\nClassification report:")
print(
    classification_report(
        y,
        predictions,
        labels=class_labels,
        zero_division=0,
        digits=3,
    )
)

print("Confusion matrix (rows=true, columns=predicted):")
print(pd.DataFrame(
    cm,
    index=[f"True {c}" for c in class_labels],
    columns=[f"Predicted {c}" for c in class_labels],
))

# Save predictions for further analysis
result = df[
    ["assembly_accession", "organism_group", "antibiotic", "label"]
].copy()
result["out_of_fold_prediction"] = predictions
result.to_csv(CHART_DIR / "out_of_fold_predictions.csv", index=False)

# Save per-class metrics
report = classification_report(
    y,
    predictions,
    labels=class_labels,
    output_dict=True,
    zero_division=0,
)
pd.DataFrame(report).transpose().to_csv(
    CHART_DIR / "classification_report.csv"
)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(7, 6))
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Susceptible", "Resistant"],
)
display.plot(ax=ax, values_format="d", colorbar=False)
ax.set_title("Random Forest: Five-Fold Grouped CV")
fig.tight_layout()
fig.savefig(
    CHART_DIR / "05_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

print("\nSaved:")
print("  project2_data\\charts\\05_confusion_matrix.png")
print("  project2_data\\charts\\classification_report.csv")
print("  project2_data\\charts\\out_of_fold_predictions.csv")


