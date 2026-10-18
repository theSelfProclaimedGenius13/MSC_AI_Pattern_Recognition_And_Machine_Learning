from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "project2_data"
OUT = ROOT / "results"
CHARTS = ROOT / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Chart 1: Compare the three model variants
# --------------------------------------------------
summary = pd.read_csv(OUT / "cross_validation_summary.csv")
summary.columns = summary.columns.str.strip()

required = {
    "Model",
    "Mean_balanced_accuracy",
    "Mean_macro_F1",
}
missing = required - set(summary.columns)
if missing:
    raise ValueError(
        f"Missing columns in cross-validation summary: {missing}\n"
        f"Available columns: {summary.columns.tolist()}"
    )

metrics = ["Mean_balanced_accuracy", "Mean_macro_F1"]
labels = ["Balanced accuracy", "Macro-F1"]

x = np.arange(len(summary))
width = 0.34

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width / 2, summary[metrics[0]], width, label=labels[0])
ax.bar(x + width / 2, summary[metrics[1]], width, label=labels[1])

ax.set_title("Antibiotic Resistance Prediction: Model Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("Feature configuration")
ax.set_xticks(x)
ax.set_xticklabels(summary["Model"], rotation=12, ha="right")
ax.set_ylim(0, 1)
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(CHARTS / "01_model_comparison.png", dpi=300)
plt.close(fig)

# --------------------------------------------------
# Chart 2: Scores across the five CV folds
# --------------------------------------------------
folds = pd.read_csv(OUT / "cross_validation_folds.csv")
folds.columns = folds.columns.str.strip()

print("\nCross-validation columns:", folds.columns.tolist())

# Accommodate minor differences in column capitalization.
def find_column(frame, candidates):
    lookup = {c.lower().replace(" ", "_"): c for c in frame.columns}
    for candidate in candidates:
        key = candidate.lower().replace(" ", "_")
        if key in lookup:
            return lookup[key]
    raise ValueError(
        f"Could not find any of {candidates}. "
        f"Available columns: {frame.columns.tolist()}"
    )

model_col = find_column(folds, ["Model"])
fold_col = find_column(folds, ["Fold"])
balanced_col = find_column(
    folds, ["Balanced_accuracy", "Balanced Accuracy"]
)
f1_col = find_column(folds, ["Macro_F1", "Macro F1"])

fig, ax = plt.subplots(figsize=(10, 6))

for model, group in folds.groupby(model_col):
    group = group.sort_values(fold_col)
    ax.plot(
        group[fold_col],
        group[balanced_col],
        marker="o",
        linewidth=2,
        label=model,
    )

ax.set_title("Balanced Accuracy Across Five Grouped CV Folds")
ax.set_xlabel("Cross-validation fold")
ax.set_ylabel("Balanced accuracy")
ax.set_xticks(sorted(folds[fold_col].unique()))
ax.set_ylim(0, 1)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(CHARTS / "02_cross_validation_folds.png", dpi=300)
plt.close(fig)

# --------------------------------------------------
# Chart 3: Per organism-antibiotic combination
# --------------------------------------------------
combos = pd.read_csv(OUT / "combination_results.csv")
combos.columns = combos.columns.str.strip()

print("\nCombination-results columns:", combos.columns.tolist())

combo_score_col = find_column(
    combos, ["balanced_accuracy", "Balanced_accuracy",
             "Balanced Accuracy"]
)

# Construct readable labels from available metadata.
organism_col = find_column(
    combos, ["organism_group", "organism", "Organism"]
)
antibiotic_col = find_column(
    combos, ["antibiotic", "Antibiotic"]
)

combos["Combination"] = (
    combos[organism_col].astype(str)
    + "\n"
    + combos[antibiotic_col].astype(str)
)
combos = combos.sort_values(combo_score_col, ascending=True)

fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.barh(combos["Combination"], combos[combo_score_col])

ax.set_title("Balanced Accuracy by Organism–Antibiotic Combination")
ax.set_xlabel("Balanced accuracy")
ax.set_ylabel("Organism and antibiotic")
ax.set_xlim(0, 1)
ax.grid(axis="x", alpha=0.3)

for bar, value in zip(bars, combos[combo_score_col]):
    ax.text(
        min(value + 0.015, 0.95),
        bar.get_y() + bar.get_height() / 2,
        f"{value:.3f}",
        va="center",
        fontsize=9,
    )

fig.tight_layout()
fig.savefig(CHARTS / "03_combination_performance.png", dpi=300)
plt.close(fig)

print("\nAll charts saved to:", CHARTS.resolve())
for path in sorted(CHARTS.glob("*.png")):
    print(f"  {path.name} ({path.stat().st_size:,} bytes)")


