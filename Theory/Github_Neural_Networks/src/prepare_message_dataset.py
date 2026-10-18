from pathlib import Path

import pandas as pd


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ENRICHED_FILE = PROCESSED_DIR / "commit_dataset_enriched.csv"
RAW_COMMITS_FILE = RAW_DIR / "commits.csv"

OUTPUT_FILE = PROCESSED_DIR / "commit_dataset_with_messages.csv"


# =========================================================
# Load datasets
# =========================================================

print("=" * 70)
print("PREPARING MESSAGE DATASET")
print("=" * 70)

print("\nLoading enriched dataset...")

enriched = pd.read_csv(ENRICHED_FILE)

print(f"Enriched rows: {len(enriched)}")


print("\nLoading raw commit data...")

raw = pd.read_csv(
    RAW_COMMITS_FILE,
    usecols=[
        "commit_sha",
        "subject",
        "message"
    ]
)

print(f"Raw commit rows: {len(raw)}")


# =========================================================
# Check commit SHA uniqueness
# =========================================================

print("\nChecking commit SHA uniqueness...")

print(
    f"Unique enriched SHAs: "
    f"{enriched['commit_sha'].nunique()}"
)

print(
    f"Unique raw SHAs: "
    f"{raw['commit_sha'].nunique()}"
)


# =========================================================
# Remove duplicate text columns if necessary
# =========================================================

raw = raw.drop_duplicates(
    subset="commit_sha"
)


# =========================================================
# Merge
# =========================================================

print("\nMerging commit text using commit_sha...")

merged = enriched.merge(
    raw,
    on="commit_sha",
    how="left",
    validate="one_to_one"
)


# =========================================================
# Check merge
# =========================================================

print("\nMerge completed.")

print(f"Rows after merge: {len(merged)}")

missing_subject = merged["subject"].isna().sum()
missing_message = merged["message"].isna().sum()

print(
    f"Missing subjects: {missing_subject}"
)

print(
    f"Missing messages: {missing_message}"
)


# =========================================================
# Check for unexpected row changes
# =========================================================

if len(merged) != len(enriched):

    raise ValueError(
        "ERROR: Merge changed the number of rows."
    )


if missing_subject > 0:

    raise ValueError(
        "ERROR: Some commits do not have a subject."
    )


# =========================================================
# Save
# =========================================================

merged.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# Final information
# =========================================================

print("\n" + "=" * 70)
print("MESSAGE DATASET CREATED")
print("=" * 70)

print(f"Output: {OUTPUT_FILE}")

print(f"\nRows    : {len(merged)}")
print(f"Columns : {len(merged.columns)}")

print("\nNew text columns:")
print("  subject")
print("  message")

print("\nDataset is ready for Experiment 2.")
print("=" * 70)
