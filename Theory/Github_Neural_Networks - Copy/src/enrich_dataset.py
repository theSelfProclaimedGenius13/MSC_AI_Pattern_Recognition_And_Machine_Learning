from pathlib import Path

import pandas as pd

from config import RAW_DATA_DIR


# ============================================================
# PATHS
# ============================================================

PROCESSED_DATA_DIR = (
    RAW_DATA_DIR.parent / "processed"
)

COMMIT_DATA_FILE = (
    PROCESSED_DATA_DIR / "commit_dataset.csv"
)

COMMIT_FILES_FILE = (
    RAW_DATA_DIR / "commit_files.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR / "commit_dataset_enriched.csv"
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def enrich_dataset():

    print("=" * 70)
    print("ENRICHING COMMIT DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading commit dataset...")

    commits = pd.read_csv(
        COMMIT_DATA_FILE
    )

    print(
        f"Commits loaded: {len(commits):,}"
    )

    print("\nLoading file-level dataset...")

    files = pd.read_csv(
        COMMIT_FILES_FILE
    )

    print(
        f"File records loaded: {len(files):,}"
    )

    # ========================================================
    # FILE-TYPE FEATURES
    # ========================================================

    print("\nCreating file-type features...")

    # --------------------------------------------------------
    # Directory groups
    # --------------------------------------------------------

    files["is_source"] = (
        files["top_level_directory"] == "source"
    ).astype(int)

    files["is_tests"] = (
        files["top_level_directory"] == "tests"
    ).astype(int)

    files["is_scripts"] = (
        files["top_level_directory"] == "scripts"
    ).astype(int)

    files["is_intern"] = (
        files["top_level_directory"] == "intern"
    ).astype(int)

    # --------------------------------------------------------
    # Extension groups
    # --------------------------------------------------------

    files["is_cpp"] = (
        files["extension"].isin(
            [".cpp", ".cc", ".cxx"]
        )
    ).astype(int)

    files["is_header"] = (
        files["extension"].isin(
            [".h", ".hh", ".hpp", ".hxx"]
        )
    ).astype(int)

    files["is_python"] = (
        files["extension"] == ".py"
    ).astype(int)

    files["is_shader"] = (
        files["extension"].isin(
            [".glsl", ".vert", ".frag"]
        )
    ).astype(int)

    # ========================================================
    # CHANGE TYPE FEATURES
    # ========================================================

    print("Creating change-type features...")

    files["is_modified"] = (
        files["change_type"] == "M"
    ).astype(int)

    files["is_added"] = (
        files["change_type"] == "A"
    ).astype(int)

    files["is_deleted"] = (
        files["change_type"] == "D"
    ).astype(int)

    files["is_renamed"] = (
        files["change_type"] == "R"
    ).astype(int)

    # ========================================================
    # AGGREGATE BY COMMIT
    # ========================================================

    print("Aggregating file information by commit...")

    grouped = files.groupby(
        "commit_sha"
    )

    features = grouped.agg(
        source_files=("is_source", "sum"),

        test_files=("is_tests", "sum"),

        script_files=("is_scripts", "sum"),

        intern_files=("is_intern", "sum"),

        cpp_files=("is_cpp", "sum"),

        header_files=("is_header", "sum"),

        python_files=("is_python", "sum"),

        shader_files=("is_shader", "sum"),

        modified_files=("is_modified", "sum"),

        added_files=("is_added", "sum"),

        deleted_files=("is_deleted", "sum"),

        renamed_files=("is_renamed", "sum"),

        average_file_depth=(
            "directory_depth",
            "mean"
        ),

        max_file_depth=(
            "directory_depth",
            "max"
        ),

        file_unique_extensions=(
            "extension",
            "nunique"
        ),

        file_unique_directories=(
            "top_level_directory",
            "nunique"
        ),
    ).reset_index()

    # ========================================================
    # RATIO FEATURES
    # ========================================================

    print("Creating ratio features...")

    total_files = (
        grouped.size()
        .reset_index(name="total_files")
    )

    features = features.merge(
        total_files,
        on="commit_sha",
        how="left"
    )

    denominator = (
        features["total_files"]
        .replace(0, 1)
    )

    features["source_file_ratio"] = (
        features["source_files"]
        / denominator
    )

    features["test_file_ratio"] = (
        features["test_files"]
        / denominator
    )

    features["cpp_file_ratio"] = (
        features["cpp_files"]
        / denominator
    )

    features["modified_file_ratio"] = (
        features["modified_files"]
        / denominator
    )

    features["added_file_ratio"] = (
        features["added_files"]
        / denominator
    )

    features["deleted_file_ratio"] = (
        features["deleted_files"]
        / denominator
    )

    # ========================================================
    # SELECT ONLY NEW FEATURES
    # ========================================================

    new_feature_columns = [
        "commit_sha",

        "source_files",
        "test_files",
        "script_files",
        "intern_files",

        "cpp_files",
        "header_files",
        "python_files",
        "shader_files",

        "modified_files",
        "added_files",
        "deleted_files",
        "renamed_files",

        "average_file_depth",
        "max_file_depth",

        "file_unique_extensions",
        "file_unique_directories",

        "source_file_ratio",
        "test_file_ratio",
        "cpp_file_ratio",
        "modified_file_ratio",
        "added_file_ratio",
        "deleted_file_ratio",
    ]

    features = features[
        new_feature_columns
    ]

    # ========================================================
    # MERGE
    # ========================================================

    print("Merging engineered features...")

    enriched = commits.merge(
        features,
        on="commit_sha",
        how="left"
    )

    # ========================================================
    # MISSING VALUES
    # ========================================================

    engineered_columns = [
        column
        for column in new_feature_columns
        if column != "commit_sha"
    ]

    enriched[engineered_columns] = (
        enriched[engineered_columns]
        .fillna(0)
    )

    # ========================================================
    # SAVE
    # ========================================================

    enriched.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("ENRICHMENT COMPLETE")
    print("=" * 70)

    print(
        f"Rows    : {len(enriched):,}"
    )

    print(
        f"Columns : {len(enriched.columns):,}"
    )

    print("\nNew engineered features:")

    for column in engineered_columns:
        print(f"  - {column}")

    print("\nOutput:")
    print(OUTPUT_FILE.resolve())

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    enrich_dataset()
