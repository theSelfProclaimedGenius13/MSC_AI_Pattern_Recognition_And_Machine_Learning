import pandas as pd

from config import DATA_DIR


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = (
    DATA_DIR
    / "processed"
    / "commit_dataset_enriched.csv"
)

OUTPUT_DIR = DATA_DIR / "processed"

TRAIN_FILE = OUTPUT_DIR / "train.csv"
VALIDATION_FILE = OUTPUT_DIR / "validation.csv"
TEST_FILE = OUTPUT_DIR / "test.csv"


# ============================================================
# Features
# ============================================================

FEATURE_COLUMNS = [
    "files_changed",
    "files_added",
    "files_deleted",
    "files_modified",
    "files_renamed",

    "unique_extensions",
    "unique_directories",
    "average_directory_depth",

    "source_files",
    "test_files",
    "script_files",
    "cpp_files",
    "header_files",
    "python_files",
    "shader_files",

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

TARGET_COLUMN = "future_bug_fix"

DATE_COLUMN = "committed_datetime"


# ============================================================
# Main
# ============================================================

def prepare_dataset():

    print("=" * 70)
    print("PREPARING CHRONOLOGICAL DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print(f"\nLoaded rows    : {len(df):,}")
    print(f"Loaded columns : {len(df.columns)}")

    # --------------------------------------------------------
    # Convert date
    # --------------------------------------------------------

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        errors="coerce",
        utc=True
    )

    # Remove invalid dates
    df = df.dropna(
        subset=[DATE_COLUMN]
    ).copy()

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        DATE_COLUMN
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Verify required columns
    # --------------------------------------------------------

    required_columns = (
        FEATURE_COLUMNS
        + [TARGET_COLUMN, DATE_COLUMN]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_columns)
        )

    # --------------------------------------------------------
    # Select only required columns
    #
    # Keep date temporarily so we can report the ranges.
    # It will NOT be used as a model feature.
    # --------------------------------------------------------

    dataset = df[
        FEATURE_COLUMNS
        + [TARGET_COLUMN, DATE_COLUMN]
    ].copy()

    # --------------------------------------------------------
    # Calculate split sizes
    # --------------------------------------------------------

    total_rows = len(dataset)

    train_size = int(
        total_rows * 0.70
    )

    validation_size = int(
        total_rows * 0.15
    )

    test_size = (
        total_rows
        - train_size
        - validation_size
    )

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    train = dataset.iloc[
        :train_size
    ].copy()

    validation = dataset.iloc[
        train_size:
        train_size + validation_size
    ].copy()

    test = dataset.iloc[
        train_size + validation_size:
    ].copy()

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    train.to_csv(
        TRAIN_FILE,
        index=False
    )

    validation.to_csv(
        VALIDATION_FILE,
        index=False
    )

    test.to_csv(
        TEST_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Helper function
    # --------------------------------------------------------

    def print_split_info(
        name,
        data
    ):

        positive = int(
            data[TARGET_COLUMN].sum()
        )

        negative = (
            len(data)
            - positive
        )

        positive_percentage = (
            positive / len(data) * 100
        )

        print()
        print("-" * 70)
        print(f"{name}")
        print("-" * 70)

        print(
            f"Rows           : {len(data):,}"
        )

        print(
            f"Features       : {len(FEATURE_COLUMNS)}"
        )

        print(
            f"Start date     : {data[DATE_COLUMN].min()}"
        )

        print(
            f"End date       : {data[DATE_COLUMN].max()}"
        )

        print(
            f"Positive (1)   : {positive:,}"
        )

        print(
            f"Negative (0)   : {negative:,}"
        )

        print(
            f"Positive %     : {positive_percentage:.2f}%"
        )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print_split_info(
        "TRAINING SET",
        train
    )

    print_split_info(
        "VALIDATION SET",
        validation
    )

    print_split_info(
        "TEST SET",
        test
    )

    # --------------------------------------------------------
    # Verify chronological separation
    # --------------------------------------------------------

    chronological = (
        train[DATE_COLUMN].max()
        <=
        validation[DATE_COLUMN].min()
        and
        validation[DATE_COLUMN].max()
        <=
        test[DATE_COLUMN].min()
)

    print()
    print("=" * 70)
    print("CHRONOLOGICAL CHECK")
    print("=" * 70)

    print(
        f"Train < Validation < Test : "
        f"{chronological}"
    )

    if not chronological:
        raise ValueError(
            "Chronological split verification failed."
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 70)

    print(
        f"\nTraining   : {TRAIN_FILE}"
    )

    print(
        f"Validation : {VALIDATION_FILE}"
    )

    print(
        f"Test       : {TEST_FILE}"
    )


if __name__ == "__main__":
    prepare_dataset()
