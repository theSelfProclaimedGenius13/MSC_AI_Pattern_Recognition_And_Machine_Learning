from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from scipy.sparse import hstack
from torch.utils.data import TensorDataset, DataLoader


# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

INPUT_FILE = (
    PROCESSED_DIR /
    "commit_dataset_with_messages.csv"
)

MODEL_FILE = (
    PROCESSED_DIR /
    "neural_network_2b.pth"
)

VECTORIZER_FILE = (
    PROCESSED_DIR /
    "tfidf_vectorizer_2b.pkl"
)

SCALER_FILE = (
    PROCESSED_DIR /
    "message_feature_scaler_2b.pkl"
)

HISTORY_FILE = (
    PROCESSED_DIR /
    "message_nn_2b_training_history.csv"
)

METRICS_FILE = (
    PROCESSED_DIR /
    "message_nn_2b_metrics.json"
)


# =========================================================
# Reproducibility
# =========================================================

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# =========================================================
# Structural features
# =========================================================

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


# =========================================================
# Load dataset
# =========================================================

print("=" * 70)
print("EXPERIMENT 2B")
print("REGULARIZED STRUCTURAL + COMMIT SUBJECT MODEL")
print("=" * 70)

print("\nLoading dataset:")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

print(f"Total commits: {len(df)}")


# =========================================================
# Chronological ordering
# =========================================================

df["committed_datetime"] = pd.to_datetime(
    df["committed_datetime"],
    utc=True
)

df = (
    df.sort_values("committed_datetime")
    .reset_index(drop=True)
)


# =========================================================
# Chronological 70/15/15 split
# =========================================================

n = len(df)

train_end = int(n * 0.70)

validation_end = int(n * 0.85)

train_df = df.iloc[:train_end].copy()

validation_df = df.iloc[
    train_end:validation_end
].copy()

test_df = df.iloc[
    validation_end:
].copy()


print("\nChronological split:")

print(f"Train      : {len(train_df)}")
print(f"Validation : {len(validation_df)}")
print(f"Test       : {len(test_df)}")


# =========================================================
# Targets
# =========================================================

y_train = (
    train_df["future_bug_fix"]
    .astype(np.float32)
    .values
)

y_validation = (
    validation_df["future_bug_fix"]
    .astype(np.float32)
    .values
)

y_test = (
    test_df["future_bug_fix"]
    .astype(np.float32)
    .values
)


# =========================================================
# Structural features
# =========================================================

X_train_structural = (
    train_df[FEATURE_COLUMNS]
    .copy()
)

X_validation_structural = (
    validation_df[FEATURE_COLUMNS]
    .copy()
)

X_test_structural = (
    test_df[FEATURE_COLUMNS]
    .copy()
)


# =========================================================
# Standardize structural features
#
# Fit ONLY on training data.
# =========================================================

scaler = StandardScaler()

X_train_structural = scaler.fit_transform(
    X_train_structural
)

X_validation_structural = scaler.transform(
    X_validation_structural
)

X_test_structural = scaler.transform(
    X_test_structural
)


# =========================================================
# Commit subjects
# =========================================================

train_subjects = (
    train_df["subject"]
    .fillna("")
    .astype(str)
    .values
)

validation_subjects = (
    validation_df["subject"]
    .fillna("")
    .astype(str)
    .values
)

test_subjects = (
    test_df["subject"]
    .fillna("")
    .astype(str)
    .values
)


# =========================================================
# TF-IDF
#
# Reduced vocabulary compared with Experiment 2A.
# =========================================================

print("\n" + "=" * 70)
print("TF-IDF")
print("=" * 70)

vectorizer = TfidfVectorizer(

    lowercase=True,

    strip_accents="unicode",

    ngram_range=(1, 2),

    min_df=3,

    max_features=800,

    sublinear_tf=True
)


print("\nFitting TF-IDF on training subjects...")

X_train_text = vectorizer.fit_transform(
    train_subjects
)

X_validation_text = vectorizer.transform(
    validation_subjects
)

X_test_text = vectorizer.transform(
    test_subjects
)


print(
    f"Vocabulary size: "
    f"{len(vectorizer.vocabulary_)}"
)

print(
    f"Training text shape: "
    f"{X_train_text.shape}"
)


# =========================================================
# Combine features
# =========================================================

X_train = hstack([
    X_train_structural,
    X_train_text
])

X_validation = hstack([
    X_validation_structural,
    X_validation_text
])

X_test = hstack([
    X_test_structural,
    X_test_text
])


# Convert sparse matrices to dense
X_train = X_train.toarray().astype(
    np.float32
)

X_validation = X_validation.toarray().astype(
    np.float32
)

X_test = X_test.toarray().astype(
    np.float32
)


print("\nCombined feature shapes:")

print(f"Train      : {X_train.shape}")
print(f"Validation : {X_validation.shape}")
print(f"Test       : {X_test.shape}")


# =========================================================
# Save preprocessing objects
# =========================================================

with open(
    VECTORIZER_FILE,
    "wb"
) as file:

    pickle.dump(
        vectorizer,
        file
    )


with open(
    SCALER_FILE,
    "wb"
) as file:

    pickle.dump(
        scaler,
        file
    )


# =========================================================
# PyTorch tensors
# =========================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32
).reshape(-1, 1)


X_validation_tensor = torch.tensor(
    X_validation,
    dtype=torch.float32
)

y_validation_tensor = torch.tensor(
    y_validation,
    dtype=torch.float32
).reshape(-1, 1)


X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.float32
).reshape(-1, 1)


# =========================================================
# DataLoaders
# =========================================================

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

validation_dataset = TensorDataset(
    X_validation_tensor,
    y_validation_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)


train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# =========================================================
# Neural Network
# =========================================================

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
            )
        )


    def forward(self, x):

        return self.network(x)


input_size = X_train.shape[1]

model = NeuralNetwork(
    input_size
)


# =========================================================
# Device
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = model.to(device)


print("\n" + "=" * 70)
print("MODEL")
print("=" * 70)

print(f"Input features : {input_size}")
print(f"Device         : {device}")

if torch.cuda.is_available():

    print(
        "GPU            : "
        f"{torch.cuda.get_device_name(0)}"
    )


# =========================================================
# Loss and optimizer
# =========================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0005,
    weight_decay=1e-3
)


# =========================================================
# Evaluation
# =========================================================

def evaluate(loader):

    model.eval()

    all_probabilities = []

    all_predictions = []

    all_targets = []

    total_loss = 0

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)

            y_batch = y_batch.to(device)

            logits = model(
                X_batch
            )

            loss = criterion(
                logits,
                y_batch
            )

            total_loss += (
                loss.item()
                * len(X_batch)
            )

            probabilities = (
                torch.sigmoid(logits)
                .cpu()
                .numpy()
                .ravel()
            )

            predictions = (
                probabilities >= 0.5
            ).astype(int)

            targets = (
                y_batch
                .cpu()
                .numpy()
                .ravel()
                .astype(int)
            )

            all_probabilities.extend(
                probabilities
            )

            all_predictions.extend(
                predictions
            )

            all_targets.extend(
                targets
            )


    all_probabilities = np.array(
        all_probabilities
    )

    all_predictions = np.array(
        all_predictions
    )

    all_targets = np.array(
        all_targets
    )


    loss = (
        total_loss /
        len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_targets,
        all_predictions
    )

    precision = precision_score(
        all_targets,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_targets,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_targets,
        all_predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        all_targets,
        all_probabilities
    )


    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }


# =========================================================
# Training
# =========================================================

MAX_EPOCHS = 100

PATIENCE = 15

best_validation_loss = float(
    "inf"
)

patience_counter = 0

history = []


print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)


for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    model.train()

    total_training_loss = 0


    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)

        y_batch = y_batch.to(device)


        optimizer.zero_grad()


        logits = model(
            X_batch
        )


        loss = criterion(
            logits,
            y_batch
        )


        loss.backward()


        optimizer.step()


        total_training_loss += (
            loss.item()
            * len(X_batch)
        )


    train_loss = (
        total_training_loss /
        len(train_dataset)
    )


    validation_metrics = evaluate(
        validation_loader
    )


    history.append({

        "epoch": epoch,

        "train_loss": train_loss,

        "validation_loss":
            validation_metrics["loss"],

        "validation_accuracy":
            validation_metrics["accuracy"],

        "validation_precision":
            validation_metrics["precision"],

        "validation_recall":
            validation_metrics["recall"],

        "validation_f1":
            validation_metrics["f1"],

        "validation_roc_auc":
            validation_metrics["roc_auc"]
    })


    print(
        f"Epoch {epoch:3d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: "
        f"{validation_metrics['loss']:.4f} | "
        f"Val Acc: "
        f"{validation_metrics['accuracy']:.4f} | "
        f"Val F1: "
        f"{validation_metrics['f1']:.4f} | "
        f"Val AUC: "
        f"{validation_metrics['roc_auc']:.4f}"
    )


    # -----------------------------------------------------
    # Early stopping
    # -----------------------------------------------------

    if (
        validation_metrics["loss"]
        < best_validation_loss
    ):

        best_validation_loss = (
            validation_metrics["loss"]
        )

        patience_counter = 0

        torch.save(
            model.state_dict(),
            MODEL_FILE
        )

    else:

        patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                f"\nEarly stopping at "
                f"epoch {epoch}."
            )

            break


# =========================================================
# Load best model
# =========================================================

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=device
    )
)


# =========================================================
# Final evaluation
# =========================================================

validation_metrics = evaluate(
    validation_loader
)

test_metrics = evaluate(
    test_loader
)


# =========================================================
# Print results
# =========================================================

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)


print("\nValidation:")

print(
    f"Loss      : "
    f"{validation_metrics['loss']:.4f}"
)

print(
    f"Accuracy  : "
    f"{validation_metrics['accuracy']:.4f}"
)

print(
    f"Precision : "
    f"{validation_metrics['precision']:.4f}"
)

print(
    f"Recall    : "
    f"{validation_metrics['recall']:.4f}"
)

print(
    f"F1        : "
    f"{validation_metrics['f1']:.4f}"
)

print(
    f"ROC-AUC   : "
    f"{validation_metrics['roc_auc']:.4f}"
)


print("\nTest:")

print(
    f"Loss      : "
    f"{test_metrics['loss']:.4f}"
)

print(
    f"Accuracy  : "
    f"{test_metrics['accuracy']:.4f}"
)

print(
    f"Precision : "
    f"{test_metrics['precision']:.4f}"
)

print(
    f"Recall    : "
    f"{test_metrics['recall']:.4f}"
)

print(
    f"F1        : "
    f"{test_metrics['f1']:.4f}"
)

print(
    f"ROC-AUC   : "
    f"{test_metrics['roc_auc']:.4f}"
)


# =========================================================
# Save history
# =========================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    HISTORY_FILE,
    index=False
)


# =========================================================
# Save metrics
# =========================================================

metrics = {

    "experiment":
        "2B",

    "structural_features":
        len(FEATURE_COLUMNS),

    "tfidf_features":
        len(vectorizer.vocabulary_),

    "total_features":
        int(input_size),

    "train_samples":
        len(train_df),

    "validation_samples":
        len(validation_df),

    "test_samples":
        len(test_df),

    "validation_accuracy":
        validation_metrics["accuracy"],

    "validation_precision":
        validation_metrics["precision"],

    "validation_recall":
        validation_metrics["recall"],

    "validation_f1":
        validation_metrics["f1"],

    "validation_roc_auc":
        validation_metrics["roc_auc"],

    "test_accuracy":
        test_metrics["accuracy"],

    "test_precision":
        test_metrics["precision"],

    "test_recall":
        test_metrics["recall"],

    "test_f1":
        test_metrics["f1"],

    "test_roc_auc":
        test_metrics["roc_auc"]
}


with open(
    METRICS_FILE,
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# =========================================================
# Done
# =========================================================

print("\nSaved model:")
print(MODEL_FILE)

print("\nSaved training history:")
print(HISTORY_FILE)

print("\nSaved metrics:")
print(METRICS_FILE)

print("\n" + "=" * 70)
print("EXPERIMENT 2B COMPLETE")
print("=" * 70)
