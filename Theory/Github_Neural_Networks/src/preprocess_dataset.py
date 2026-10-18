import json
import pickle

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from config import DATA_DIR


# ============================================================
# Configuration
# ============================================================

PROCESSED_DIR = DATA_DIR / "processed"

TRAIN_FILE = PROCESSED_DIR / "train.csv"
VALIDATION_FILE = PROCESSED_DIR / "validation.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"

OUTPUT_FILE = PROCESSED_DIR / "preprocessed_data.npz"
SCALER_FILE = PROCESSED_DIR / "scaler.pkl"
FEATURES_FILE = PROCESSED_DIR / "feature_names.json"


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


# Features representing counts rather than ratios/depths.
# These are strongly right-skewed, so log1p is applied.
COUNT_FEATURES = [
    "files_changed",
    "files_added",
    "files_deleted",
    "files_modified",
    "files_renamed",

    "unique_extensions",
    "unique_directories",

    "source_files",
    "test_files",
    "script_files",
    "cpp_files",
    "header_files",
    "python_files",
    "shader_files",

    "file_unique_extensions",
    "file_unique_directories",
]


# ============================================================
# Helper
# ============================================================

def prepare_features(df):
    """
    Extract model features and apply log1p to count features.
    """

    X = df[FEATURE_COLUMNS].copy()

    # --------------------------------------------------------
    # Log-transform highly skewed count features
    # --------------------------------------------------------

    for column in COUNT_FEATURES:
        X[column] = np.log1p(X[column])

    return X.to_numpy(dtype=np.float64)


# ============================================================
# Main
# ============================================================

def preprocess_dataset():

    print("=" * 70)
    print("PREPROCESSING DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    train = pd.read_csv(TRAIN_FILE)
    validation = pd.read_csv(VALIDATION_FILE)
    test = pd.read_csv(TEST_FILE)

    print(f"\nTraining rows   : {len(train):,}")
    print(f"Validation rows : {len(validation):,}")
    print(f"Test rows       : {len(test):,}")

    # --------------------------------------------------------
    # Prepare X
    # --------------------------------------------------------

    X_train = prepare_features(train)
    X_validation = prepare_features(validation)
    X_test = prepare_features(test)

    # --------------------------------------------------------
    # Prepare y
    # --------------------------------------------------------

    y_train = train[TARGET_COLUMN].to_numpy(
        dtype=np.float32
    )

    y_validation = validation[TARGET_COLUMN].to_numpy(
        dtype=np.float32
    )

    y_test = test[TARGET_COLUMN].to_numpy(
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Standardization
    #
    # IMPORTANT:
    # The scaler is fitted ONLY on training data.
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_validation = scaler.transform(
        X_validation
    )

    X_test = scaler.transform(
        X_test
    )

    # --------------------------------------------------------
    # Save scaler
    # --------------------------------------------------------

    with open(
        SCALER_FILE,
        "wb"
    ) as file:

        pickle.dump(
            scaler,
            file
        )

    # --------------------------------------------------------
    # Save feature names
    # --------------------------------------------------------

    with open(
        FEATURES_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            FEATURE_COLUMNS,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Save processed arrays
    # --------------------------------------------------------

    np.savez(
        OUTPUT_FILE,

        X_train=X_train,
        y_train=y_train,

        X_validation=X_validation,
        y_validation=y_validation,

        X_test=X_test,
        y_test=y_test,
    )

    # --------------------------------------------------------
    # Print information
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"\nNumber of features : {X_train.shape[1]}"
    )

    print(
        f"X_train shape     : {X_train.shape}"
    )

    print(
        f"X_validation shape: {X_validation.shape}"
    )

    print(
        f"X_test shape      : {X_test.shape}"
    )

    print("\nTarget shapes:")

    print(
        f"y_train           : {y_train.shape}"
    )

    print(
        f"y_validation      : {y_validation.shape}"
    )

    print(
        f"y_test            : {y_test.shape}"
    )

    # --------------------------------------------------------
    # Verify training standardization
    # --------------------------------------------------------

    print("\nTraining feature statistics after scaling:")

    print(
        f"Mean absolute value : "
        f"{np.abs(X_train.mean(axis=0)).mean():.6f}"
    )

    print(
        f"Mean standard deviation : "
        f"{X_train.std(axis=0).mean():.6f}"
    )

    # --------------------------------------------------------
    # Output files
    # --------------------------------------------------------

    print("\nSaved files:")

    print(
        f"Data   : {OUTPUT_FILE}"
    )

    print(
        f"Scaler : {SCALER_FILE}"
    )

    print(
        f"Features: {FEATURES_FILE}"
    )


if __name__ == "__main__":
    preprocess_dataset()
