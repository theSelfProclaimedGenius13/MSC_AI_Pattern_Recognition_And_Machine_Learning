from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "project2_data"
RESULTS_DIR = DATA_DIR / "results"
CHARTS_DIR = DATA_DIR / "charts"
INPUT_FILE = CHARTS_DIR / "out_of_fold_predictions.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Prediction file not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)
required = ["assembly_accession", "organism_group", "antibiotic", "label", "out_of_fold_prediction"]
missing = set(required) - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

mapping = {"susceptible": 0, "resistant": 1, "0": 0, "1": 1}
for col in ["label", "out_of_fold_prediction"]:
    df[col] = df[col].astype("string").str.strip().str.lower().map(mapping)

df = df.dropna(subset=["label", "out_of_fold_prediction"]).copy()
if df.empty:
    raise ValueError("No valid labels found.")
df["label"] = df["label"].astype(int)
df["out_of_fold_prediction"] = df["out_of_fold_prediction"].astype(int)

names = {0: "Susceptible", 1: "Resistant"}
df["actual_label"] = df["label"].map(names)
df["predicted_label"] = df["out_of_fold_prediction"].map(names)
df["correct_prediction"] = df["label"].eq(df["out_of_fold_prediction"])

tp = df[(df.label == 1) & (df.out_of_fold_prediction == 1)]
tn = df[(df.label == 0) & (df.out_of_fold_prediction == 0)]
fp = df[(df.label == 0) & (df.out_of_fold_prediction == 1)].copy()
fn = df[(df.label == 1) & (df.out_of_fold_prediction == 0)].copy()

total = len(df)
correct = len(tp) + len(tn)
errors = len(fp) + len(fn)
accuracy = correct / total
error_rate = errors / total
cm = confusion_matrix(df.label, df.out_of_fold_prediction, labels=[0, 1])

print("\\n" + "=" * 60)
print("FALSE POSITIVE AND FALSE NEGATIVE ANALYSIS")
print("=" * 60)
print(f"Total records: {total}")
print(f"Correct: {correct} ({accuracy:.1%}) | Incorrect: {errors} ({error_rate:.1%})")
print(f"TP: {len(tp)} | TN: {len(tn)} | FP: {len(fp)} | FN: {len(fn)}")
print("\\nConfusion matrix (rows actual [Susceptible, Resistant]; columns predicted):")
print(cm)
print("\\nClassification report:")
print(classification_report(df.label, df.out_of_fold_prediction, labels=[0, 1],
                            target_names=["Susceptible", "Resistant"], zero_division=0))

cols = ["assembly_accession", "organism_group", "antibiotic", "actual_label", "predicted_label"]
print("\\nFALSE POSITIVE EXAMPLES (actual susceptible, predicted resistant)")
print(fp[cols].head(10).to_string(index=False) if len(fp) else "No false positives.")
print("\\nFALSE NEGATIVE EXAMPLES (actual resistant, predicted susceptible)")
print(fn[cols].head(10).to_string(index=False) if len(fn) else "No false negatives.")

fp[cols].to_csv(RESULTS_DIR / "false_positives.csv", index=False)
fn[cols].to_csv(RESULTS_DIR / "false_negatives.csv", index=False)

rows = []
for (organism, antibiotic), group in df.groupby(["organism_group", "antibiotic"], dropna=False):
    group_fp = int(((group.label == 0) & (group.out_of_fold_prediction == 1)).sum())
    group_fn = int(((group.label == 1) & (group.out_of_fold_prediction == 0)).sum())
    n = len(group)
    rows.append({
        "organism_group": organism,
        "antibiotic": antibiotic,
        "total_records": n,
        "false_positives": group_fp,
        "false_negatives": group_fn,
        "total_errors": group_fp + group_fn,
        "error_rate_percent": round(100 * (group_fp + group_fn) / n, 2),
    })

error_df = pd.DataFrame(rows).sort_values("error_rate_percent", ascending=False)
error_csv = RESULTS_DIR / "misclassification_by_combination.csv"
error_df.to_csv(error_csv, index=False)
print("\\nMISCLASSIFICATIONS BY ORGANISM AND ANTIBIOTIC")
print(error_df.to_string(index=False))

# Generate chart
plot_df = error_df.copy()
plot_df["combination"] = plot_df["organism_group"].astype(str) + " / " + plot_df["antibiotic"].astype(str)
plot_df = plot_df.sort_values("error_rate_percent", ascending=True)
