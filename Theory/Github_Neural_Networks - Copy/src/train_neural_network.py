# src/train_neural_network.py

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from config import DATA_DIR


# ============================================================
# Configuration
# ============================================================

PROCESSED_DIR = DATA_DIR / "processed"

DATA_FILE = PROCESSED_DIR / "preprocessed_data.npz"

MODEL_FILE = PROCESSED_DIR / "neural_network.pth"

HISTORY_FILE = PROCESSED_DIR / "training_history.csv"


# Neural network parameters
INPUT_SIZE = 25
HIDDEN_SIZE_1 = 32
HIDDEN_SIZE_2 = 16

LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 100

EARLY_STOPPING_PATIENCE = 15


# ============================================================
# Neural Network
# ============================================================

class CommitNeuralNetwork(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.network = nn.Sequential(

            # Input → Hidden layer 1
            nn.Linear(
                input_size,
                HIDDEN_SIZE_1
            ),

            nn.ReLU(),

            # Hidden layer 1 → Hidden layer 2
            nn.Linear(
                HIDDEN_SIZE_1,
                HIDDEN_SIZE_2
            ),

            nn.ReLU(),

            # Hidden layer 2 → Output
            nn.Linear(
                HIDDEN_SIZE_2,
                1
            )
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# Evaluation function
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    loss_function,
    device,
):

    model.eval()

    with torch.no_grad():

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32,
            device=device,
        )

        y_tensor = torch.tensor(
            y,
            dtype=torch.float32,
            device=device,
        ).view(-1, 1)

        logits = model(X_tensor)

        loss = loss_function(
            logits,
            y_tensor
        ).item()

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities >= 0.5
        ).float()

    y_true = y_tensor.cpu().numpy().flatten()

    y_pred = predictions.cpu().numpy().flatten()

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

    return (
        loss,
        accuracy,
        precision,
        recall,
        f1,
    )


# ============================================================
# Main training function
# ============================================================

def train_neural_network():

    print("=" * 70)
    print("NEURAL NETWORK TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Select device
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = torch.device("cuda")

    else:

        device = torch.device("cpu")

    print(
        f"\nDevice: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Load preprocessed data
    # --------------------------------------------------------

    data = np.load(
        DATA_FILE
    )

    X_train = data["X_train"]
    y_train = data["y_train"]

    X_validation = data["X_validation"]
    y_validation = data["y_validation"]

    X_test = data["X_test"]
    y_test = data["y_test"]

    print("\nDataset shapes:")

    print(
        f"X_train      : {X_train.shape}"
    )

    print(
        f"X_validation : {X_validation.shape}"
    )

    print(
        f"X_test       : {X_test.shape}"
    )

    # --------------------------------------------------------
    # Convert training data to PyTorch tensors
    # --------------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.float32
    ).view(-1, 1)

    # --------------------------------------------------------
    # Create DataLoader
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = CommitNeuralNetwork(
        INPUT_SIZE
    ).to(device)

    print("\nModel:")
    print(model)

    # --------------------------------------------------------
    # Loss function
    #
    # BCEWithLogitsLoss combines:
    #     Sigmoid + Binary Cross Entropy
    #
    # We therefore do NOT put Sigmoid inside the model.
    # --------------------------------------------------------

    loss_function = nn.BCEWithLogitsLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history = []

    best_validation_loss = float("inf")

    best_model_state = None

    epochs_without_improvement = 0

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):

        model.train()

        total_training_loss = 0.0

        # ----------------------------------------------------
        # Mini-batch training
        # ----------------------------------------------------

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # -----------------------------------------------
            # Clear previous gradients
            # -----------------------------------------------

            optimizer.zero_grad()

            # -----------------------------------------------
            # Forward propagation
            # -----------------------------------------------

            logits = model(X_batch)

            # -----------------------------------------------
            # Calculate loss
            # -----------------------------------------------

            loss = loss_function(
                logits,
                y_batch
            )

            # -----------------------------------------------
            # Backpropagation
            # -----------------------------------------------

            loss.backward()

            # -----------------------------------------------
            # Update weights
            # -----------------------------------------------

            optimizer.step()

            total_training_loss += (
                loss.item()
                * len(X_batch)
            )

        # Average training loss
        training_loss = (
            total_training_loss
            / len(train_dataset)
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        (
            validation_loss,
            validation_accuracy,
            validation_precision,
            validation_recall,
            validation_f1,
        ) = evaluate_model(
            model,
            X_validation,
            y_validation,
            loss_function,
            device,
        )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        history.append({

            "epoch": epoch,

            "training_loss":
                training_loss,

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
        })

        # ----------------------------------------------------
        # Print progress
        # ----------------------------------------------------

        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {training_loss:.4f} | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Acc: {validation_accuracy:.4f} | "
            f"Val F1: {validation_f1:.4f}"
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                f"\nEarly stopping at epoch "
                f"{epoch}."
            )

            break

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    if best_model_state is not None:

        model.load_state_dict(
            best_model_state
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    torch.save(
        model.state_dict(),
        MODEL_FILE
    )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        HISTORY_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Final validation metrics
    # --------------------------------------------------------

    (
        validation_loss,
        validation_accuracy,
        validation_precision,
        validation_recall,
        validation_f1,
    ) = evaluate_model(
        model,
        X_validation,
        y_validation,
        loss_function,
        device,
    )

    # --------------------------------------------------------
    # Final TEST metrics
    # --------------------------------------------------------

    (
        test_loss,
        test_accuracy,
        test_precision,
        test_recall,
        test_f1,
    ) = evaluate_model(
        model,
        X_test,
        y_test,
        loss_function,
        device,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print("\nValidation:")
    print(
        f"Loss      : {validation_loss:.4f}"
    )
    print(
        f"Accuracy  : {validation_accuracy:.4f}"
    )
    print(
        f"Precision : {validation_precision:.4f}"
    )
    print(
        f"Recall    : {validation_recall:.4f}"
    )
    print(
        f"F1 Score  : {validation_f1:.4f}"
    )

    print("\nTest:")
    print(
        f"Loss      : {test_loss:.4f}"
    )
    print(
        f"Accuracy  : {test_accuracy:.4f}"
    )
    print(
        f"Precision : {test_precision:.4f}"
    )
    print(
        f"Recall    : {test_recall:.4f}"
    )
    print(
        f"F1 Score  : {test_f1:.4f}"
    )

    # --------------------------------------------------------
    # Saved files
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"\nModel   : {MODEL_FILE}"
    )

    print(
        f"History : {HISTORY_FILE}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    train_neural_network()
