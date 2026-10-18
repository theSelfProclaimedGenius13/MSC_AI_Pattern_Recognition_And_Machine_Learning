from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "project2_data"
CHART_DIR = DATA_DIR / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

model_path = ROOT / "models" / "random_forest_amr_model.joblib"
dataset_path = DATA_DIR / "processed" / "model_dataset.csv"

# Load the previously trained model and dataset
model = joblib.load(model_path)
df = pd.read_csv(dataset_path, nrows=1)

# Identify DNA 4-mer features: exactly four A/C/G/T characters
kmers = [
    col for col in df.columns
    if len(col) == 4 and set(col).issubset(set("ACGT"))
]

if not kmers:
    raise ValueError("No DNA 4-mer columns found in model_dataset.csv")


# Get the Random Forest inside the fitted pipeline
forest = model.named_steps["classifier"]
preprocessor = model.named_steps["preprocess"]

# These names match the transformed features used by the forest
feature_names = preprocessor.get_feature_names_out()
importances = forest.feature_importances_

if len(feature_names) != len(importances):
    raise ValueError(
        f"Mismatch: {len(feature_names)} transformed features, "
        f"but {len(importances)} importance values."
    )

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances,
})

# Keep only DNA 4-mers, excluding one-hot encoded metadata
importance_df["Kmer"] = importance_df["Feature"].str.split(
    "__"
).str[-1]

importance_df = importance_df[
    importance_df["Kmer"].isin(kmers)
].copy()

importance_df = importance_df.rename(
    columns={"Kmer": "DNA_4mer"}
)

importance_df = importance_df[
    ["DNA_4mer", "Importance"]
].sort_values("Importance", ascending=False)

top20 = importance_df.head(20).sort_values("Importance")

if top20.empty:
    raise ValueError("No DNA 4-mer features matched the transformed features.")



# Save importance tables
importance_df.to_csv(
    CHART_DIR / "all_kmer_importances.csv", index=False
)
importance_df.head(20).to_csv(
    CHART_DIR / "top20_kmer_importances.csv", index=False
)

# Plot the top 20 DNA 4-mers
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(top20["DNA_4mer"], top20["Importance"])
ax.set_title("Top 20 DNA 4-mers by Random Forest Feature Importance")
ax.set_xlabel("Feature importance")
ax.set_ylabel("DNA 4-mer")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(
    CHART_DIR / "04_top20_kmer_importance.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

print("\nTop 20 DNA 4-mers:")
print(importance_df.head(20).to_string(index=False))
print("\nSaved chart: project2_data\\charts\\04_top20_kmer_importance.png")
print("Saved table: project2_data\\charts\\top20_kmer_importances.csv")


