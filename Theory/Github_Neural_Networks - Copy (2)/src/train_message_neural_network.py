import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

from torch.utils.data import TensorDataset, DataLoader


# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

INPUT_FILE = PROCESSED_DIR / "commit_dataset_with_messages.csv"
MODEL_FILE = PROCESSED_DIR / "neural_network_with_messages.pth"
VECTORIZER_FILE = PROCESSED_DIR / "tfidf_vectorizer.pkl"
SCALER_FILE = PROCESSED_DIR / "message_feature_scaler.pkl"
HISTORY_FILE = PROCESSED_DIR / "message_nn_training_history.csv"
METRICS_FILE = PROCESSED_DIR / "message_nn_metrics.json"


# =========================================================
# Features
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
# Reproducibility
# =========================================================

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# =========================================================
# Load dataset
# =========================================================

print("=" * 70)
print("EXPERIMENT 2")
print("STRUCTURAL FEATURES + COMMIT MESSAGE")
print("=" * 70)

print(f"\nLoading dataset:")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

print(f"Total commits available: {len(df)}")


# =========================================================
# Sort chronologically
# =========================================================

df["committed_datetime"] = pd.to_datetime(
    df["committed_datetime"],
    utc=True
)

df = df.sort_values("committed_datetime").reset_index(drop=True)


# =========================================================
# Select same 70/15/15 chronological split
# =========================================================

n = len(df)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)

train_df = df.iloc[:train_end].copy()
validation_df = df.iloc[train_end:validation_end].copy()
test_df = df.iloc[validation_end:].copy()

print("\nChronological split:")
print(f"Train      : {len(train_df)}")
print(f"Validation : {len(validation_df)}")
print(f"Test       : {len(test_df)}")

print("\nDate ranges:")

print(
    f"Train      : "
    f"{train_df['committed_datetime'].min()} -> "
    f"{train_df['committed_datetime'].max()}"
)

print(
    f"Validation : "
    f"{validation_df['committed_datetime'].min()} -> "
    f"{validation_df['committed_datetime'].max()}"
)

print(
    f"Test       : "
    f"{test_df['committed_datetime'].min()} -> "
    f"{test_df['committed_datetime'].max()}"
)


# =========================================================
# Extract structural features
# =========================================================

X_train_structural = train_df[FEATURE_COLUMNS].copy()
X_validation_structural = validation_df[FEATURE_COLUMNS].copy()
X_test_structural = test_df[FEATURE_COLUMNS].copy()

y_train = train_df["future_bug_fix"].astype(np.float32).values
y_validation = validation_df["future_bug_fix"].astype(np.float32).values
y_test = test_df["future_bug_fix"].astype(np.float32).values


# =========================================================
# Scale structural features
#
# IMPORTANT:
# Fit scaler ONLY on training data.
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
# Extract commit subjects
# =========================================================
#
# The processed dataset contains the commit subject.
# We use the subject as the text representation of the
# commit message for this experiment.
#
# This is still information available at commit time.
# =========================================================

train_messages = (
    train_df["subject"]
    .fillna("")
    .astype(str)
    .values
)

validation_messages = (
    validation_df["subject"]
    .fillna("")
    .astype(str)
    .values
)

test_messages = (
    test_df["subject"]
    .fillna("")
    .astype(str)
    .values
)

# =========================================================
# TF-IDF
#
# Vocabulary is learned ONLY from training messages.
# =========================================================

print("\n" + "=" * 70)
print("TF-IDF TEXT PROCESSING")
print("=" * 70)

vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=2,
    max_features=5000,
    sublinear_tf=True
)

print("\nFitting TF-IDF vocabulary on training messages...")

X_train_text = vectorizer.fit_transform(
    train_messages
)

print(
    f"Vocabulary size: "
    f"{len(vectorizer.vocabulary_)}"
)

print(
    f"Training text matrix: "
    f"{X_train_text.shape}"
)

X_validation_text = vectorizer.transform(
    validation_messages
)

X_test_text = vectorizer.transform(
    test_messages
)


# =========================================================
# Combine structural + text features
# =========================================================

print("\nCombining structural and text features...")

X_train_combined = hstack([
    X_train_structural,
    X_train_text
])

X_validation_combined = hstack([
    X_validation_structural,
    X_validation_text
])

X_test_combined = hstack([
    X_test_structural,
    X_test_text
])


# Convert to dense arrays for PyTorch
X_train_combined = X_train_combined.toarray().astype(
    np.float32
)

X_validation_combined = X_validation_combined.toarray().astype(
    np.float32
)

X_test_combined = X_test_combined.toarray().astype(
    np.float32
)

print(
    f"Combined training shape: "
    f"{X_train_combined.shape}"
)

print(
    f"Combined validation shape: "
    f"{X_validation_combined.shape}"
)

print(
    f"Combined test shape: "
    f"{X_test_combined.shape}"
)


# =========================================================
# Save preprocessing objects
# =========================================================

with open(VECTORIZER_FILE, "wb") as file:
    pickle.dump(vectorizer, file)

with open(SCALER_FILE, "wb") as file:
    pickle.dump(scaler, file)

print("\nSaved TF-IDF vectorizer:")
print(VECTORIZER_FILE)

print("\nSaved scaler:")
print(SCALER_FILE)


# =========================================================
# PyTorch datasets
# =========================================================

X_train_tensor = torch.tensor(
    X_train_combined,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32
).reshape(-1, 1)


X_validation_tensor = torch.tensor(
    X_validation_combined,
    dtype=torch.float32
)

y_validation_tensor = torch.tensor(
    y_validation,
    dtype=torch.float32
).reshape(-1, 1)


X_test_tensor = torch.tensor(
    X_test_combined,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.float32
).reshape(-1, 1)


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

            nn.Linear(input_size, 64),

            nn.ReLU(),

            nn.Dropout(0.30),

            nn.Linear(64, 32),

            nn.ReLU(),

            nn.Dropout(0.20),

            nn.Linear(32, 1)
        )

    def forward(self, x):

        return self.network(x)


input_size = X_train_combined.shape[1]

model = NeuralNetwork(input_size)


# =========================================================
# Device
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

print("\n" + "=" * 70)
print("NEURAL NETWORK")
print("=" * 70)

print(f"Input features : {input_size}")
print(f"Device         : {device}")

if torch.cuda.is_available():

    print(
        f"GPU            : "
        f"{torch.cuda.get_device_name(0)}"
    )


# =========================================================
# Loss and optimizer
# =========================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)


# =========================================================
# Evaluation function
# =========================================================

def evaluate(loader):

    model.eval()

    total_loss = 0

    correct = 0
    total = 0

    true_positive = 0
    false_positive = 0
    false_negative = 0

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)

            loss = criterion(
                logits,
                y_batch
            )

            total_loss += (
                loss.item() * len(X_batch)
            )

            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= 0.5
            ).float()

            correct += (
                predictions == y_batch
            ).sum().item()

            total += len(X_batch)

            true_positive += (
                (predictions == 1) &
                (y_batch == 1)
            ).sum().item()

            false_positive += (
                (predictions == 1) &
                (y_batch == 0)
            ).sum().item()

            false_negative += (
                (predictions == 0) &
                (y_batch == 1)
            ).sum().item()

    loss = total_loss / total

    accuracy = correct / total

    precision = (
        true_positive /
        (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0
    )

    recall = (
        true_positive /
        (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return (
        loss,
        accuracy,
        precision,
        recall,
        f1
    )


# =========================================================
# Training
# =========================================================

MAX_EPOCHS = 100
PATIENCE = 15

best_validation_loss = float("inf")
patience_counter = 0

history = []

print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)


for epoch in range(1, MAX_EPOCHS + 1):

    model.train()

    total_training_loss = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)

        loss = criterion(
            logits,
            y_batch
        )

        loss.backward()

        optimizer.step()

        total_training_loss += (
            loss.item() * len(X_batch)
        )

    train_loss = (
        total_training_loss /
        len(train_dataset)
    )

    (
        validation_loss,
        validation_accuracy,
        validation_precision,
        validation_recall,
        validation_f1
    ) = evaluate(validation_loader)

    history.append({

        "epoch": epoch,

        "train_loss": train_loss,

        "validation_loss": validation_loss,

        "validation_accuracy":
            validation_accuracy,

        "validation_precision":
            validation_precision,

        "validation_recall":
            validation_recall,

        "validation_f1":
            validation_f1
    })

    print(
        f"Epoch {epoch:3d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {validation_loss:.4f} | "
        f"Val Acc: {validation_accuracy:.4f} | "
        f"Val F1: {validation_f1:.4f}"
    )

    # -----------------------------------------------------
    # Early stopping
    # -----------------------------------------------------

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        patience_counter = 0

        torch.save(
            model.state_dict(),
            MODEL_FILE
        )

    else:

        patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                f"\nEarly stopping at epoch "
                f"{epoch}."
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

(
    validation_loss,
    validation_accuracy,
    validation_precision,
    validation_recall,
    validation_f1
) = evaluate(validation_loader)


(
    test_loss,
    test_accuracy,
    test_precision,
    test_recall,
    test_f1
) = evaluate(test_loader)


# =========================================================
# Results
# =========================================================

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print("\nValidation:")
print(f"Loss      : {validation_loss:.4f}")
print(f"Accuracy  : {validation_accuracy:.4f}")
print(f"Precision : {validation_precision:.4f}")
print(f"Recall    : {validation_recall:.4f}")
print(f"F1        : {validation_f1:.4f}")

print("\nTest:")
print(f"Loss      : {test_loss:.4f}")
print(f"Accuracy  : {test_accuracy:.4f}")
print(f"Precision : {test_precision:.4f}")
print(f"Recall    : {test_recall:.4f}")
print(f"F1        : {test_f1:.4f}")


# =========================================================
# Save training history
# =========================================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    HISTORY_FILE,
    index=False
)


# =========================================================
# Save metrics
# =========================================================

metrics = {

    "input_features":
        int(input_size),

    "tfidf_features":
        int(len(vectorizer.vocabulary_)),

    "structural_features":
        len(FEATURE_COLUMNS),

    "train_samples":
        len(train_df),

    "validation_samples":
        len(validation_df),

    "test_samples":
        len(test_df),

    "validation_accuracy":
        validation_accuracy,

    "validation_precision":
        validation_precision,

    "validation_recall":
        validation_recall,

    "validation_f1":
        validation_f1,

    "test_accuracy":
        test_accuracy,

    "test_precision":
        test_precision,

    "test_recall":
        test_recall,

    "test_f1":
        test_f1
}

with open(METRICS_FILE, "w") as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# =========================================================
# Finished
# =========================================================

print("\nSaved model:")
print(MODEL_FILE)

print("\nSaved training history:")
print(HISTORY_FILE)

print("\nSaved metrics:")
print(METRICS_FILE)

print("\n" + "=" * 70)
print("EXPERIMENT 2 COMPLETE")
print("=" * 70)
