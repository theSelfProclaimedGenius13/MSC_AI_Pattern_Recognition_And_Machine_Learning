# final_evaluation.py
#
# Final evaluation of all models for the GitHub commit
# future bug-fix involvement prediction project.
#
# Models:
#   1. Majority-class baseline
#   2. Logistic Regression
#   3. Random Forest
#   4. Structural Neural Network
#   5. Structural + Subject TF-IDF Neural Network
#   6. Structural + Full Message TF-IDF Neural Network
#
# All models use the same chronological train/validation/test split.
#
# Outputs:
#   final_model_comparison.csv
#   final_metrics_summary.csv
#   final_roc_curves.png
#   final_confusion_matrix.png
#   final_precision_recall_f1.png
#
# NOTE:
# The neural networks use their saved trained checkpoints.
# Logistic Regression and Random Forest are retrained using
# the same training data and fixed random_state=42.


from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

import torch
import torch.nn as nn


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"

PROCESSED_DIR = DATA_DIR / "processed"


DATASET_PATH = (
    PROCESSED_DIR
    / "commit_dataset_with_messages.csv"
)


# Output files

COMPARISON_PATH = (
    PROCESSED_DIR
    / "final_model_comparison.csv"
)


SUMMARY_PATH = (
    PROCESSED_DIR
    / "final_metrics_summary.csv"
)


ROC_PATH = (
    PROCESSED_DIR
    / "final_roc_curves.png"
)


CONFUSION_PATH = (
    PROCESSED_DIR
    / "final_confusion_matrix.png"
)


PRF_PATH = (
    PROCESSED_DIR
    / "final_precision_recall_f1.png"
)


# ============================================================
# DATA SPLIT
# ============================================================

TRAIN_SIZE = 2963

VALIDATION_SIZE = 635

TEST_SIZE = 636


# ============================================================
# FEATURES
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
# NEURAL NETWORK ARCHITECTURES
# ============================================================

class StructuralNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                25,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                16
            ),

            nn.ReLU(),

            nn.Linear(
                16,
                1
            ),
        )


    def forward(self, x):

        return self.network(x)


class TextNN(nn.Module):

    def __init__(
        self,
        input_size
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_size,
                32
            ),

            nn.ReLU(),

            nn.Dropout(
                0.40
            ),

            nn.Linear(
                32,
                16
            ),

            nn.ReLU(),

            nn.Dropout(
                0.30
            ),

            nn.Linear(
                16,
                1
            ),
        )


    def forward(self, x):

        return self.network(x)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)

print("FINAL MODEL EVALUATION")

print("=" * 70)

print(
    f"Device: {DEVICE}"
)

if torch.cuda.is_available():

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATASET_PATH
)


df["committed_datetime"] = pd.to_datetime(
    df["committed_datetime"],
    utc=True,
)


df = df.sort_values(
    "committed_datetime"
).reset_index(
    drop=True
)


EXPECTED_TOTAL = (
    TRAIN_SIZE
    + VALIDATION_SIZE
    + TEST_SIZE
)


if len(df) != EXPECTED_TOTAL:

    raise ValueError(
        f"Expected {EXPECTED_TOTAL} rows "
        f"but found {len(df)}."
    )


# ============================================================
# SPLIT
# ============================================================

train_df = df.iloc[
    :TRAIN_SIZE
].copy()


validation_df = df.iloc[
    TRAIN_SIZE:
    TRAIN_SIZE + VALIDATION_SIZE
].copy()


test_df = df.iloc[
    TRAIN_SIZE + VALIDATION_SIZE:
].copy()


print("\nSplit:")

print(
    f"Train      : {len(train_df)}"
)

print(
    f"Validation : {len(validation_df)}"
)

print(
    f"Test       : {len(test_df)}"
)


# ============================================================
# STRUCTURAL PREPROCESSING
# ============================================================

print(
    "\nPreparing structural features..."
)


X_train_struct = (
    train_df[FEATURE_COLUMNS]
    .apply(pd.to_numeric)
    .copy()
)


X_validation_struct = (
    validation_df[FEATURE_COLUMNS]
    .apply(pd.to_numeric)
    .copy()
)


X_test_struct = (
    test_df[FEATURE_COLUMNS]
    .apply(pd.to_numeric)
    .copy()
)


# ------------------------------------------------------------
# log1p
# ------------------------------------------------------------

for column in COUNT_FEATURES:

    X_train_struct[column] = np.log1p(
        X_train_struct[column]
    )

    X_validation_struct[column] = np.log1p(
        X_validation_struct[column]
    )

    X_test_struct[column] = np.log1p(
        X_test_struct[column]
    )


# ------------------------------------------------------------
# StandardScaler
# ------------------------------------------------------------

from sklearn.preprocessing import StandardScaler


structural_scaler = StandardScaler()


X_train_struct_scaled = (
    structural_scaler.fit_transform(
        X_train_struct
    )
)


X_validation_struct_scaled = (
    structural_scaler.transform(
        X_validation_struct
    )
)


X_test_struct_scaled = (
    structural_scaler.transform(
        X_test_struct
    )
)


# ============================================================
# TARGET
# ============================================================

y_train = (
    train_df[TARGET_COLUMN]
    .astype(int)
    .values
)


y_validation = (
    validation_df[TARGET_COLUMN]
    .astype(int)
    .values
)


y_test = (
    test_df[TARGET_COLUMN]
    .astype(int)
    .values
)


# ============================================================
# RESULT STORAGE
# ============================================================

results = {}

probabilities = {}

predictions = {}


# ============================================================
# EVALUATION HELPER
# ============================================================

def calculate_metrics(
    name,
    y_true,
    y_pred,
    y_prob,
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_true,
        y_prob
    )


    results[name] = {

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "ROC-AUC": auc,
    }


    probabilities[name] = y_prob

    predictions[name] = y_pred


# ============================================================
# 1. MAJORITY BASELINE
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "1. MAJORITY BASELINE"
)

print(
    "=" * 70
)


majority_class = int(
    pd.Series(y_train)
    .mode()[0]
)


majority_predictions = np.full(
    len(y_test),
    majority_class
)


# A constant probability is not useful for ROC-AUC.
# Therefore, ROC-AUC is left as NaN for the majority baseline.

majority_accuracy = accuracy_score(
    y_test,
    majority_predictions
)


majority_precision = precision_score(
    y_test,
    majority_predictions,
    zero_division=0
)


majority_recall = recall_score(
    y_test,
    majority_predictions,
    zero_division=0
)


majority_f1 = f1_score(
    y_test,
    majority_predictions,
    zero_division=0
)


results["Majority Baseline"] = {

    "Model":
        "Majority Baseline",

    "Accuracy":
        majority_accuracy,

    "Precision":
        majority_precision,

    "Recall":
        majority_recall,

    "F1":
        majority_f1,

    "ROC-AUC":
        np.nan,
}


predictions["Majority Baseline"] = (
    majority_predictions
)


print(
    f"Majority class: "
    f"{majority_class}"
)

print(
    f"Accuracy: "
    f"{majority_accuracy:.4f}"
)


# ============================================================
# 2. LOGISTIC REGRESSION
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "2. LOGISTIC REGRESSION"
)

print(
    "=" * 70
)


logistic_model = LogisticRegression(

    max_iter=2000,

    random_state=42,
)


logistic_model.fit(
    X_train_struct_scaled,
    y_train
)


logistic_prob = (
    logistic_model
    .predict_proba(
        X_test_struct_scaled
    )[:, 1]
)


logistic_pred = (
    logistic_prob >= 0.5
).astype(int)


calculate_metrics(
    "Logistic Regression",
    y_test,
    logistic_pred,
    logistic_prob
)


print(
    f"Accuracy: "
    f"{results['Logistic Regression']['Accuracy']:.4f}"
)

print(
    f"F1: "
    f"{results['Logistic Regression']['F1']:.4f}"
)

print(
    f"ROC-AUC: "
    f"{results['Logistic Regression']['ROC-AUC']:.4f}"
)


# ============================================================
# 3. RANDOM FOREST
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "3. RANDOM FOREST"
)

print(
    "=" * 70
)


random_forest = RandomForestClassifier(

    n_estimators=300,

    random_state=42,

    n_jobs=-1,
)


random_forest.fit(
    X_train_struct_scaled,
    y_train
)


rf_prob = (
    random_forest
    .predict_proba(
        X_test_struct_scaled
    )[:, 1]
)


rf_pred = (
    rf_prob >= 0.5
).astype(int)


calculate_metrics(
    "Random Forest",
    y_test,
    rf_pred,
    rf_prob
)


print(
    f"Accuracy: "
    f"{results['Random Forest']['Accuracy']:.4f}"
)

print(
    f"F1: "
    f"{results['Random Forest']['F1']:.4f}"
)

print(
    f"ROC-AUC: "
    f"{results['Random Forest']['ROC-AUC']:.4f}"
)


# ============================================================
# 4. STRUCTURAL NEURAL NETWORK
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "4. STRUCTURAL NEURAL NETWORK"
)

print(
    "=" * 70
)


STRUCTURAL_MODEL_PATH = (
    PROCESSED_DIR
    / "neural_network.pth"
)


structural_model = (
    StructuralNN()
    .to(DEVICE)
)


structural_model.load_state_dict(
    torch.load(
        STRUCTURAL_MODEL_PATH,
        map_location=DEVICE,
    )
)


structural_model.eval()


with torch.no_grad():

    X_tensor = torch.tensor(
        X_test_struct_scaled,
        dtype=torch.float32,
        device=DEVICE,
    )


    structural_logits = (
        structural_model(
            X_tensor
        )
    )


    structural_prob = (
        torch.sigmoid(
            structural_logits
        )
        .cpu()
        .numpy()
        .ravel()
    )


structural_pred = (
    structural_prob >= 0.5
).astype(int)


calculate_metrics(
    "Structural Neural Network",
    y_test,
    structural_pred,
    structural_prob
)


print(
    f"Accuracy: "
    f"{results['Structural Neural Network']['Accuracy']:.4f}"
)

print(
    f"F1: "
    f"{results['Structural Neural Network']['F1']:.4f}"
)

print(
    f"ROC-AUC: "
    f"{results['Structural Neural Network']['ROC-AUC']:.4f}"
)


# ============================================================
# HELPER FOR TEXT MODELS
# ============================================================

def load_text_model(
    model_path,
    vectorizer_path,
    scaler_path,
    text_column,
    model_name,
):

    print(
        "\n"
        + "=" * 70
    )

    print(model_name)

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Load scaler
    # --------------------------------------------------------

    with open(
        scaler_path,
        "rb",
    ) as file:

        scaler = pickle.load(
            file
        )


    # --------------------------------------------------------
    # Load TF-IDF vectorizer
    # --------------------------------------------------------

    with open(
        vectorizer_path,
        "rb",
    ) as file:

        vectorizer = pickle.load(
            file
        )


    # --------------------------------------------------------
    # Transform structural data
    # --------------------------------------------------------

    X_train_struct_local = (
        train_df[FEATURE_COLUMNS]
        .apply(pd.to_numeric)
        .copy()
    )


    X_test_struct_local = (
        test_df[FEATURE_COLUMNS]
        .apply(pd.to_numeric)
        .copy()
    )


    for column in COUNT_FEATURES:

        X_train_struct_local[column] = (
            np.log1p(
                X_train_struct_local[column]
            )
        )


        X_test_struct_local[column] = (
            np.log1p(
                X_test_struct_local[column]
            )
        )


    X_train_struct_local = (
        scaler.transform(
            X_train_struct_local
        )
    )


    X_test_struct_local = (
        scaler.transform(
            X_test_struct_local
        )
    )


    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    train_text = (
        train_df[text_column]
        .fillna("")
        .astype(str)
    )


    test_text = (
        test_df[text_column]
        .fillna("")
        .astype(str)
    )


    X_train_text = (
        vectorizer.transform(
            train_text
        )
    )


    X_test_text = (
        vectorizer.transform(
            test_text
        )
    )


    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    X_train_combined = np.hstack(
        [
            X_train_struct_local,
            X_train_text.toarray(),
        ]
    )


    X_test_combined = np.hstack(
        [
            X_test_struct_local,
            X_test_text.toarray(),
        ]
    )


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = (
        TextNN(
            X_train_combined.shape[1]
        )
        .to(DEVICE)
    )


    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE,
        )
    )


    model.eval()


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    with torch.no_grad():

        X_tensor = torch.tensor(
            X_test_combined,
            dtype=torch.float32,
            device=DEVICE,
        )


        logits = model(
            X_tensor
        )


        probabilities = (
            torch.sigmoid(
                logits
            )
            .cpu()
            .numpy()
            .ravel()
        )


    predictions_local = (
        probabilities >= 0.5
    ).astype(int)


    calculate_metrics(
        model_name,
        y_test,
        predictions_local,
        probabilities
    )


    print(
        f"Features: "
        f"{X_train_combined.shape[1]}"
    )

    print(
        f"Accuracy: "
        f"{results[model_name]['Accuracy']:.4f}"
    )

    print(
        f"F1: "
        f"{results[model_name]['F1']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{results[model_name]['ROC-AUC']:.4f}"
    )


# ============================================================
# 5. SUBJECT NEURAL NETWORK
# ============================================================

load_text_model(

    model_path=(
        PROCESSED_DIR
        / "neural_network_2b_fair.pth"
    ),

    vectorizer_path=(
        PROCESSED_DIR
        / "tfidf_vectorizer_2b_fair.pkl"
    ),

    scaler_path=(
        PROCESSED_DIR
        / "message_feature_scaler_2b_fair.pkl"
    ),

    text_column="subject",

    model_name=(
        "Subject TF-IDF Neural Network"
    ),
)


# ============================================================
# 6. FULL MESSAGE NEURAL NETWORK
# ============================================================

load_text_model(

    model_path=(
        PROCESSED_DIR
        / "neural_network_full_message_fair.pth"
    ),

    vectorizer_path=(
        PROCESSED_DIR
        / "tfidf_vectorizer_full_message_fair.pkl"
    ),

    scaler_path=(
        PROCESSED_DIR
        / "full_message_feature_scaler_fair.pkl"
    ),

    text_column="message",

    model_name=(
        "Full Message TF-IDF Neural Network"
    ),
)


# ============================================================
# MODEL COMPARISON TABLE
# ============================================================

comparison_df = pd.DataFrame(
    results.values()
)


comparison_df = comparison_df[
    [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
    ]
]


comparison_df.to_csv(
    COMPARISON_PATH,
    index=False,
)


# ============================================================
# PRINT COMPARISON
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "FINAL MODEL COMPARISON"
)

print(
    "=" * 70
)


print(
    comparison_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)


# ============================================================
# SUMMARY TABLE FOR PPT
# ============================================================

summary_df = comparison_df.copy()


for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC",
]:

    summary_df[column] = (
        summary_df[column] * 100
    )


summary_df.to_csv(
    SUMMARY_PATH,
    index=False,
)


# ============================================================
# ROC CURVES
# ============================================================

print(
    "\nGenerating ROC curves..."
)


plt.figure(
    figsize=(10, 7)
)


for name, probability in probabilities.items():

    if name == "Majority Baseline":
        continue


    fpr, tpr, _ = roc_curve(
        y_test,
        probability
    )


    auc = roc_auc_score(
        y_test,
        probability
    )


    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=(
            f"{name} "
            f"(AUC={auc:.3f})"
        ),
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    label="Random classifier",
)


plt.xlabel(
    "False Positive Rate"
)


plt.ylabel(
    "True Positive Rate"
)


plt.title(
    "ROC Curves - Model Comparison"
)


plt.legend(
    loc="lower right",
    fontsize=8,
)


plt.grid(
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    ROC_PATH,
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# CONFUSION MATRIX
# ============================================================

print(
    "Generating confusion matrix..."
)


FINAL_MODEL_NAME = (
    "Full Message TF-IDF Neural Network"
)


final_predictions = predictions[
    FINAL_MODEL_NAME
]


cm = confusion_matrix(
    y_test,
    final_predictions
)


plt.figure(
    figsize=(7, 6)
)


plt.imshow(
    cm,
    interpolation="nearest"
)


plt.title(
    "Confusion Matrix - Final Neural Network"
)


plt.colorbar()


tick_marks = np.arange(2)


plt.xticks(
    tick_marks,
    [
        "No Future Bug-Fix",
        "Future Bug-Fix",
    ],
    rotation=20,
)


plt.yticks(
    tick_marks,
    [
        "No Future Bug-Fix",
        "Future Bug-Fix",
    ],
)


threshold = cm.max() / 2


for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=14,
        )


plt.xlabel(
    "Predicted Label"
)


plt.ylabel(
    "Actual Label"
)


plt.tight_layout()


plt.savefig(
    CONFUSION_PATH,
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# PRECISION / RECALL / F1 COMPARISON
# ============================================================

print(
    "Generating Precision/Recall/F1 chart..."
)


plot_df = comparison_df[
    comparison_df["Model"]
    != "Majority Baseline"
].copy()


x = np.arange(
    len(plot_df)
)


width = 0.25


plt.figure(
    figsize=(12, 7)
)


plt.bar(
    x - width,
    plot_df["Precision"],
    width,
    label="Precision",
)


plt.bar(
    x,
    plot_df["Recall"],
    width,
    label="Recall",
)


plt.bar(
    x + width,
    plot_df["F1"],
    width,
    label="F1",
)


plt.xticks(
    x,
    plot_df["Model"],
    rotation=20,
    ha="right",
)


plt.ylabel(
    "Score"
)


plt.title(
    "Precision, Recall and F1 - Model Comparison"
)


plt.ylim(
    0,
    1
)


plt.legend()


plt.grid(
    axis="y",
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    PRF_PATH,
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# FINAL RESULT
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "FINAL MODEL"
)

print(
    "=" * 70
)


final_row = comparison_df[
    comparison_df["Model"]
    == FINAL_MODEL_NAME
].iloc[0]


print(
    f"Model     : {FINAL_MODEL_NAME}"
)

print(
    f"Accuracy  : "
    f"{final_row['Accuracy']:.4f}"
)

print(
    f"Precision : "
    f"{final_row['Precision']:.4f}"
)

print(
    f"Recall    : "
    f"{final_row['Recall']:.4f}"
)

print(
    f"F1        : "
    f"{final_row['F1']:.4f}"
)

print(
    f"ROC-AUC   : "
    f"{final_row['ROC-AUC']:.4f}"
)


# ============================================================
# FINAL OUTPUT FILES
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "SAVED FILES"
)

print(
    "=" * 70
)

print(
    COMPARISON_PATH
)

print(
    SUMMARY_PATH
)

print(
    ROC_PATH
)

print(
    CONFUSION_PATH
)

print(
    PRF_PATH
)


print(
    "\nFinal evaluation complete."
)
