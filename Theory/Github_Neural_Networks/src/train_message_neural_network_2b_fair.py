# train_message_neural_network_2b_fair.py
#
# Neural Network:
#   25 structural commit features
#   + TF-IDF commit subject features
#
# Structural preprocessing:
#   count features -> log1p()
#   StandardScaler fitted ONLY on training data
#
# Text preprocessing:
#   TF-IDF fitted ONLY on training subjects
#
# Split:
#   Same chronological 70/15/15 split used by the project
#
# Target:
#   future_bug_fix


from pathlib import Path
import pickle
import json

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

DATASET_PATH = (
    PROCESSED_DIR
    / "commit_dataset_with_messages.csv"
)

MODEL_PATH = (
    PROCESSED_DIR
    / "neural_network_2b_fair.pth"
)

VECTORIZER_PATH = (
    PROCESSED_DIR
    / "tfidf_vectorizer_2b_fair.pkl"
)

SCALER_PATH = (
    PROCESSED_DIR
    / "message_feature_scaler_2b_fair.pkl"
)

HISTORY_PATH = (
    PROCESSED_DIR
    / "message_nn_2b_fair_training_history.csv"
)

METRICS_PATH = (
    PROCESSED_DIR
    / "message_nn_2b_fair_metrics.json"
)


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


# ============================================================
# COUNT FEATURES
# ============================================================

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
# TRAINING SETTINGS
# ============================================================

TRAIN_SIZE = 2963
VALIDATION_SIZE = 635
TEST_SIZE = 636

BATCH_SIZE = 64

MAX_EPOCHS = 100

PATIENCE = 15

LEARNING_RATE = 0.0005

WEIGHT_DECAY = 1e-3


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("FAIR SUBJECT-TEXT NEURAL NETWORK")
print("=" * 70)

print(f"Device: {DEVICE}")


if torch.cuda.is_available():

    print(
        f"GPU   : "
        f"{torch.cuda.get_device_name(0)}"
    )


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATASET_PATH
)

print(
    f"Dataset rows: {len(df)}"
)


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = (
    FEATURE_COLUMNS
    + [
        TARGET_COLUMN,
        "committed_datetime",
        "subject",
    ]
)


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        "The following required columns "
        "are missing:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# CHRONOLOGICAL SORT
# ============================================================

print(
    "\nSorting commits chronologically..."
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


# ============================================================
# VERIFY EXPECTED DATASET SIZE
# ============================================================

EXPECTED_TOTAL = (
    TRAIN_SIZE
    + VALIDATION_SIZE
    + TEST_SIZE
)


if len(df) != EXPECTED_TOTAL:

    raise ValueError(
        f"Expected {EXPECTED_TOTAL} rows "
        f"but found {len(df)} rows."
    )


# ============================================================
# CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
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


print("\nDataset split:")

print(
    f"Train        : {len(train_df)}"
)

print(
    f"Validation   : {len(validation_df)}"
)

print(
    f"Test         : {len(test_df)}"
)


# ============================================================
# VERIFY TARGET DISTRIBUTION
# ============================================================

print("\nTarget distribution:")


for name, data in [
    ("Train", train_df),
    ("Validation", validation_df),
    ("Test", test_df),
]:

    counts = (
        data[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    print(
        f"{name}: "
        f"0 = {counts.get(0, 0)}, "
        f"1 = {counts.get(1, 0)}"
    )


# ============================================================
# SUBJECT TEXT
# ============================================================

train_subjects = (
    train_df["subject"]
    .fillna("")
    .astype(str)
)


validation_subjects = (
    validation_df["subject"]
    .fillna("")
    .astype(str)
)


test_subjects = (
    test_df["subject"]
    .fillna("")
    .astype(str)
)


print("\nChecking commit subjects...")

print(
    f"Train empty subjects: "
    f"{(train_subjects.str.strip() == '').sum()}"
)

print(
    f"Validation empty subjects: "
    f"{(validation_subjects.str.strip() == '').sum()}"
)

print(
    f"Test empty subjects: "
    f"{(test_subjects.str.strip() == '').sum()}"
)


# ============================================================
# STRUCTURAL FEATURES
# ============================================================

print(
    "\nPreparing structural features..."
)


X_train_struct = (
    train_df[FEATURE_COLUMNS]
    .copy()
)


X_validation_struct = (
    validation_df[FEATURE_COLUMNS]
    .copy()
)


X_test_struct = (
    test_df[FEATURE_COLUMNS]
    .copy()
)


# Convert everything to numeric

X_train_struct = (
    X_train_struct.apply(
        pd.to_numeric
    )
)


X_validation_struct = (
    X_validation_struct.apply(
        pd.to_numeric
    )
)


X_test_struct = (
    X_test_struct.apply(
        pd.to_numeric
    )
)


# ============================================================
# LOG1P TRANSFORMATION
# ============================================================

print(
    "\nApplying log1p to count features..."
)


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


# ============================================================
# STANDARDIZATION
# ============================================================

print(
    "\nApplying StandardScaler..."
)

scaler = StandardScaler()


# IMPORTANT:
# Fit ONLY on training data.

X_train_struct_scaled = (
    scaler.fit_transform(
        X_train_struct
    )
)


X_validation_struct_scaled = (
    scaler.transform(
        X_validation_struct
    )
)


X_test_struct_scaled = (
    scaler.transform(
        X_test_struct
    )
)


print(
    "Structural feature count: "
    f"{X_train_struct_scaled.shape[1]}"
)


# ============================================================
# TF-IDF
# ============================================================

print(
    "\nCreating TF-IDF features..."
)


vectorizer = TfidfVectorizer(

    lowercase=True,

    ngram_range=(1, 2),

    min_df=3,

    max_features=800,

    sublinear_tf=True,
)


# IMPORTANT:
# Fit ONLY on training subjects.

X_train_text = (
    vectorizer.fit_transform(
        train_subjects
    )
)


X_validation_text = (
    vectorizer.transform(
        validation_subjects
    )
)


X_test_text = (
    vectorizer.transform(
        test_subjects
    )
)


print(
    "TF-IDF feature count: "
    f"{X_train_text.shape[1]}"
)


# ============================================================
# COMBINE STRUCTURAL + TEXT FEATURES
# ============================================================

print(
    "\nCombining structural and text features..."
)


X_train = np.hstack(
    [
        X_train_struct_scaled,
        X_train_text.toarray(),
    ]
)


X_validation = np.hstack(
    [
        X_validation_struct_scaled,
        X_validation_text.toarray(),
    ]
)


X_test = np.hstack(
    [
        X_test_struct_scaled,
        X_test_text.toarray(),
    ]
)


# ============================================================
# TARGET
# ============================================================

y_train = (
    train_df[TARGET_COLUMN]
    .astype(np.float32)
    .values
)


y_validation = (
    validation_df[TARGET_COLUMN]
    .astype(np.float32)
    .values
)


y_test = (
    test_df[TARGET_COLUMN]
    .astype(np.float32)
    .values
)


print("\nFinal feature dimensions:")

print(
    f"Structural features : "
    f"{X_train_struct_scaled.shape[1]}"
)

print(
    f"TF-IDF features     : "
    f"{X_train_text.shape[1]}"
)

print(
    f"Total features      : "
    f"{X_train.shape[1]}"
)

print(
    f"Train shape         : "
    f"{X_train.shape}"
)

print(
    f"Validation shape    : "
    f"{X_validation.shape}"
)

print(
    f"Test shape          : "
    f"{X_test.shape}"
)


# ============================================================
# PYTORCH TENSORS
# ============================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32,
)


y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32,
).reshape(-1, 1)


X_validation_tensor = torch.tensor(
    X_validation,
    dtype=torch.float32,
)


y_validation_tensor = torch.tensor(
    y_validation,
    dtype=torch.float32,
).reshape(-1, 1)


X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32,
)


y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.float32,
).reshape(-1, 1)


# ============================================================
# TRAIN DATALOADER
# ============================================================

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor,
)


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,
)


# ============================================================
# NEURAL NETWORK
# ============================================================

class NeuralNetwork(nn.Module):

    def __init__(self, input_size):

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


model = NeuralNetwork(
    input_size=X_train.shape[1]
).to(DEVICE)


print(
    "\nNeural network architecture:"
)

print(model)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.BCEWithLogitsLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(
    model,
    X,
    y,
):

    model.eval()


    with torch.no_grad():

        X = X.to(DEVICE)

        y = y.to(DEVICE)


        logits = model(X)


        loss = criterion(
            logits,
            y,
        )


        probabilities = torch.sigmoid(
            logits
        )


        predictions = (
            probabilities >= 0.5
        ).float()


    y_true = (
        y.cpu()
        .numpy()
        .ravel()
    )


    y_pred = (
        predictions.cpu()
        .numpy()
        .ravel()
    )


    y_prob = (
        probabilities.cpu()
        .numpy()
        .ravel()
    )


    accuracy = accuracy_score(
        y_true,
        y_pred,
    )


    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )


    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )


    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )


    roc_auc = roc_auc_score(
        y_true,
        y_prob,
    )


    return (
        loss.item(),
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
    )


# ============================================================
# TRAINING
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "TRAINING"
)

print(
    "=" * 70
)


best_validation_loss = float(
    "inf"
)


epochs_without_improvement = 0


history = []


for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    model.train()


    epoch_loss = 0.0


    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(
            DEVICE
        )

        y_batch = y_batch.to(
            DEVICE
        )


        optimizer.zero_grad()


        logits = model(
            X_batch
        )


        loss = criterion(
            logits,
            y_batch,
        )


        loss.backward()


        optimizer.step()


        epoch_loss += (
            loss.item()
            * len(X_batch)
        )


    train_loss = (
        epoch_loss
        / len(train_dataset)
    )


    (
        validation_loss,
        validation_accuracy,
        validation_precision,
        validation_recall,
        validation_f1,
        validation_auc,
    ) = evaluate(

        model,

        X_validation_tensor,

        y_validation_tensor,
    )


    history.append({

        "epoch": epoch,

        "train_loss": train_loss,

        "validation_loss":
            validation_loss,

        "validation_accuracy":
            validation_accuracy,

        "validation_precision":
            validation_precision,

        "validation_recall":
            validation_recall,

        "validation_f1":
            validation_f1,

        "validation_roc_auc":
            validation_auc,
    })


    print(

        f"Epoch {epoch:3d} | "

        f"Train Loss: "
        f"{train_loss:.4f} | "

        f"Val Loss: "
        f"{validation_loss:.4f} | "

        f"Val Acc: "
        f"{validation_accuracy:.4f} | "

        f"Val F1: "
        f"{validation_f1:.4f} | "

        f"Val AUC: "
        f"{validation_auc:.4f}"
    )


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if (
        validation_loss
        < best_validation_loss
    ):

        best_validation_loss = (
            validation_loss
        )

        epochs_without_improvement = 0


        torch.save(
            model.state_dict(),
            MODEL_PATH,
        )


    else:

        epochs_without_improvement += 1


        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "\nEarly stopping at "
                f"epoch {epoch}."
            )

            break


# ============================================================
# LOAD BEST MODEL
# ============================================================

print(
    "\nLoading best model..."
)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )
)


# ============================================================
# FINAL VALIDATION RESULTS
# ============================================================

(
    validation_loss,
    validation_accuracy,
    validation_precision,
    validation_recall,
    validation_f1,
    validation_auc,
) = evaluate(

    model,

    X_validation_tensor,

    y_validation_tensor,
)


# ============================================================
# FINAL TEST RESULTS
# ============================================================

(
    test_loss,
    test_accuracy,
    test_precision,
    test_recall,
    test_f1,
    test_auc,
) = evaluate(

    model,

    X_test_tensor,

    y_test_tensor,
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)


history_df.to_csv(
    HISTORY_PATH,
    index=False,
)


# ============================================================
# SAVE TF-IDF VECTORIZER
# ============================================================

with open(
    VECTORIZER_PATH,
    "wb",
) as file:

    pickle.dump(
        vectorizer,
        file,
    )


# ============================================================
# SAVE SCALER
# ============================================================

with open(
    SCALER_PATH,
    "wb",
) as file:

    pickle.dump(
        scaler,
        file,
    )


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "model":
        "Neural Network - "
        "Structural + Subject TF-IDF",

    "structural_features":
        len(FEATURE_COLUMNS),

    "tfidf_features":
        X_train_text.shape[1],

    "total_features":
        X_train.shape[1],

    "validation": {

        "loss":
            validation_loss,

        "accuracy":
            validation_accuracy,

        "precision":
            validation_precision,

        "recall":
            validation_recall,

        "f1":
            validation_f1,

        "roc_auc":
            validation_auc,
    },

    "test": {

        "loss":
            test_loss,

        "accuracy":
            test_accuracy,

        "precision":
            test_precision,

        "recall":
            test_recall,

        "f1":
            test_f1,

        "roc_auc":
            test_auc,
    },

    "preprocessing": {

        "count_transformation":
            "log1p",

        "scaler":
            "StandardScaler",

        "tfidf_ngram_range":
            [1, 2],

        "tfidf_min_df":
            3,

        "tfidf_max_features":
            800,

        "sublinear_tf":
            True,
    },

    "training": {

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "max_epochs":
            MAX_EPOCHS,

        "early_stopping_patience":
            PATIENCE,
    },

    "architecture": [

        f"Linear({X_train.shape[1]}, 32)",

        "ReLU",

        "Dropout(0.40)",

        "Linear(32, 16)",

        "ReLU",

        "Dropout(0.30)",

        "Linear(16, 1)",
    ],
}


with open(
    METRICS_PATH,
    "w",
) as file:

    json.dump(
        metrics,
        file,
        indent=4,
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "FINAL VALIDATION RESULTS"
)

print(
    "=" * 70
)

print(
    f"Loss      : "
    f"{validation_loss:.4f}"
)

print(
    f"Accuracy  : "
    f"{validation_accuracy:.4f}"
)

print(
    f"Precision : "
    f"{validation_precision:.4f}"
)

print(
    f"Recall    : "
    f"{validation_recall:.4f}"
)

print(
    f"F1        : "
    f"{validation_f1:.4f}"
)

print(
    f"ROC-AUC   : "
    f"{validation_auc:.4f}"
)


print(
    "\n"
    + "=" * 70
)

print(
    "FINAL TEST RESULTS"
)

print(
    "=" * 70
)

print(
    f"Loss      : "
    f"{test_loss:.4f}"
)

print(
    f"Accuracy  : "
    f"{test_accuracy:.4f}"
)

print(
    f"Precision : "
    f"{test_precision:.4f}"
)

print(
    f"Recall    : "
    f"{test_recall:.4f}"
)

print(
    f"F1        : "
    f"{test_f1:.4f}"
)

print(
    f"ROC-AUC   : "
    f"{test_auc:.4f}"
)


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
    MODEL_PATH
)

print(
    VECTORIZER_PATH
)

print(
    SCALER_PATH
)

print(
    HISTORY_PATH
)

print(
    METRICS_PATH
)


print(
    "\nDone."
)
