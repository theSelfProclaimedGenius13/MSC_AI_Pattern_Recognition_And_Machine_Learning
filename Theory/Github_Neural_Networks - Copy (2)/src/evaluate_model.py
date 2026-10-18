# src/evaluate_model.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

from config import DATA_DIR


# ============================================================
# Configuration
# ============================================================

PROCESSED_DIR = DATA_DIR / "processed"

DATA_FILE = PROCESSED_DIR / "preprocessed_data.npz"
MODEL_FILE = PROCESSED_DIR / "neural_network.pth"

CONFUSION_MATRIX_FILE = (
    PROCESSED_DIR / "confusion_matrix.png"
)

ROC_CURVE_FILE = (
    PROCESSED_DIR / "roc_curve.png"
)


# ============================================================
# Neural Network
# ============================================================

class CommitNeuralNetwork(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_size,
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
            )
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# Main
# ============================================================

def evaluate_model():

    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = torch.device("cuda")

    else:

        device = torch.device("cpu")

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    data = np.load(
        DATA_FILE
    )

    X_test = data["X_test"]
    y_test = data["y_test"]

    print(
        f"Test samples: {len(X_test):,}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = CommitNeuralNetwork(
        input_size=X_test.shape[1]
    ).to(device)

    model.load_state_dict(
        torch.load(
            MODEL_FILE,
            map_location=device,
            weights_only=True,
        )
    )

    model.eval()

    # --------------------------------------------------------
    # Convert test data to tensors
    # --------------------------------------------------------

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32,
        device=device,
    )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            X_test_tensor
        )

        probabilities = torch.sigmoid(
            logits
        ).cpu().numpy().flatten()

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # Basic metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST METRICS")
    print("=" * 70)

    print(
        f"\nAccuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "No Future Bug-Fix Involvement",
            "Future Bug-Fix Involvement",
        ],
        zero_division=0,
    )

    print(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(
        "\n             Predicted"
    )

    print(
        "             0      1"
    )

    print(
        f"Actual 0   {cm[0, 0]:4d}   {cm[0, 1]:4d}"
    )

    print(
        f"Actual 1   {cm[1, 0]:4d}   {cm[1, 1]:4d}"
    )

    # --------------------------------------------------------
    # Majority-class baseline
    # --------------------------------------------------------

    majority_class = int(
        pd.Series(y_test)
        .value_counts()
        .idxmax()
    )

    baseline_predictions = np.full(
        len(y_test),
        majority_class
    )

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_predictions
    )

    print("\n" + "=" * 70)
    print("BASELINE")
    print("=" * 70)

    print(
        f"Majority class       : {majority_class}"
    )

    print(
        f"Baseline accuracy    : "
        f"{baseline_accuracy:.4f}"
    )

    print(
        f"Neural network       : "
        f"{accuracy:.4f}"
    )

    print(
        f"Improvement          : "
        f"{accuracy - baseline_accuracy:+.4f}"
    )

    # --------------------------------------------------------
    # Prediction distribution
    # --------------------------------------------------------

    predicted_zeros = np.sum(
        predictions == 0
    )

    predicted_ones = np.sum(
        predictions == 1
    )

    print("\n" + "=" * 70)
    print("PREDICTION DISTRIBUTION")
    print("=" * 70)

    print(
        f"Predicted 0 : {predicted_zeros:,}"
    )

    print(
        f"Predicted 1 : {predicted_ones:,}"
    )

    # --------------------------------------------------------
    # Confusion matrix plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    image = ax.imshow(
        cm
    )

    ax.set_title(
        "Confusion Matrix"
    )

    ax.set_xlabel(
        "Predicted Label"
    )

    ax.set_ylabel(
        "Actual Label"
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels([
        "No Future Fix",
        "Future Fix",
    ])

    ax.set_yticklabels([
        "No Future Fix",
        "Future Fix",
    ])

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=14,
            )

    fig.colorbar(image)

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_FILE,
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # ROC curve
    # --------------------------------------------------------

    false_positive_rate, true_positive_rate, _ = (
        roc_curve(
            y_test,
            probabilities
        )
    )

    roc_area = auc(
        false_positive_rate,
        true_positive_rate
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"Neural Network (AUC = {roc_area:.3f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        ROC_CURVE_FILE,
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"\nConfusion matrix:"
        f"\n{CONFUSION_MATRIX_FILE}"
    )

    print(
        f"\nROC curve:"
        f"\n{ROC_CURVE_FILE}"
    )


if __name__ == "__main__":
    evaluate_model()
