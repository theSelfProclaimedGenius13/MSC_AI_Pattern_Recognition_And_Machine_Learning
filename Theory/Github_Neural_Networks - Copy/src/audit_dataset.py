import pandas as pd

from config import DATA_DIR


# ============================================================
# Configuration
# ============================================================

DATASET_FILE = (
    DATA_DIR
    / "processed"
    / "commit_dataset_enriched.csv"
)


# ============================================================
# Main audit
# ============================================================

def audit_dataset():

    print("=" * 70)
    print("DATASET AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(DATASET_FILE)

    print(f"\nRows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    # --------------------------------------------------------
    # Column information
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COLUMNS")
    print("=" * 70)

    for i, column in enumerate(df.columns, start=1):
        print(f"{i:2}. {column:<30} {df[column].dtype}")

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MISSING VALUES")
    print("=" * 70)

    missing = df.isna().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:
        print("No missing values found.")
    else:
        print(missing.sort_values(ascending=False))

    # --------------------------------------------------------
    # Duplicate commits
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DUPLICATES")
    print("=" * 70)

    duplicate_sha = df["commit_sha"].duplicated().sum()

    print(f"Duplicate commit SHAs : {duplicate_sha}")

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TARGET DISTRIBUTION")
    print("=" * 70)

    target_counts = df["future_bug_fix"].value_counts().sort_index()

    for value, count in target_counts.items():

        percentage = (
            count / len(df) * 100
        )

        label = (
            "Negative (0)"
            if value == 0
            else "Positive (1)"
        )

        print(
            f"{label:<18}: "
            f"{count:>6,} "
            f"({percentage:6.2f}%)"
        )

    # --------------------------------------------------------
    # Date information
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATE RANGE")
    print("=" * 70)

    df["committed_datetime"] = pd.to_datetime(
        df["committed_datetime"],
        errors="coerce"
    )

    print(
        "Earliest commit :",
        df["committed_datetime"].min()
    )

    print(
        "Latest commit   :",
        df["committed_datetime"].max()
    )

    # --------------------------------------------------------
    # Check chronological ordering
    # --------------------------------------------------------

    is_sorted = df["committed_datetime"].is_monotonic_increasing

    print(
        "\nChronologically sorted:",
        is_sorted
    )

    # --------------------------------------------------------
    # Numerical feature statistics
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("NUMERICAL FEATURES")
    print("=" * 70)

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    print(
        df[numeric_columns]
        .describe()
        .T
        .to_string()
    )

    # --------------------------------------------------------
    # Constant features
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CONSTANT FEATURES")
    print("=" * 70)

    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]

    if constant_columns:
        for column in constant_columns:
            print(column)
    else:
        print("No constant features found.")

    # --------------------------------------------------------
    # High-cardinality categorical columns
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CATEGORICAL / HIGH-CARDINALITY FEATURES")
    print("=" * 70)

    categorical_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in categorical_columns:

        unique_count = df[column].nunique()

        print(
            f"{column:<30} "
            f"unique values: {unique_count:,}"
        )

    # --------------------------------------------------------
    # Target vs numerical features
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE MEANS BY TARGET")
    print("=" * 70)

    feature_columns = [
        column
        for column in numeric_columns
        if column != "future_bug_fix"
    ]

    grouped = (
        df.groupby("future_bug_fix")[feature_columns]
        .mean()
        .T
    )

    print(grouped.to_string())

    # --------------------------------------------------------
    # Feature correlations
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CORRELATION WITH TARGET")
    print("=" * 70)

    correlations = (
        df[numeric_columns]
        .corr()["future_bug_fix"]
        .drop("future_bug_fix")
        .sort_values(
            key=lambda x: x.abs(),
            ascending=False
        )
    )

    print(correlations.to_string())

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    audit_dataset()
