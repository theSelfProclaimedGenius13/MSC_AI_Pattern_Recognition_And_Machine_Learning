import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROCESSED_DIR = "Theory/Github_Neural_Networks/data/processed"

DATA_FILE = f"{PROCESSED_DIR}/preprocessed_data.npz"
OUTPUT_FILE = f"{PROCESSED_DIR}/baseline_comparison.csv"


# ---------------------------------------------------------
# Load preprocessed data
# ---------------------------------------------------------

data = np.load(DATA_FILE)

X_train = data["X_train"]
y_train = data["y_train"]

X_validation = data["X_validation"]
y_validation = data["y_validation"]

X_test = data["X_test"]
y_test = data["y_test"]


print("=" * 70)
print("BASELINE MODEL COMPARISON")
print("=" * 70)

print(f"Training samples   : {len(X_train)}")
print(f"Validation samples : {len(X_validation)}")
print(f"Test samples       : {len(X_test)}")
print(f"Features           : {X_train.shape[1]}")


# ---------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------

def evaluate_model(name, model, X, y):
    predictions = model.predict(X)

    # Probability of class 1
    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)
    roc_auc = roc_auc_score(y, probabilities)

    matrix = confusion_matrix(y, predictions)

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc,
    }


# ---------------------------------------------------------
# Model 1: Logistic Regression
# ---------------------------------------------------------

print("\nTraining Logistic Regression...")

logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

logistic_model.fit(X_train, y_train)


# Validation evaluation
print("\nValidation performance:")

evaluate_model(
    "Logistic Regression - Validation",
    logistic_model,
    X_validation,
    y_validation
)


# Test evaluation
print("\nTest performance:")

logistic_results = evaluate_model(
    "Logistic Regression - Test",
    logistic_model,
    X_test,
    y_test
)


# ---------------------------------------------------------
# Model 2: Random Forest
# ---------------------------------------------------------

print("\nTraining Random Forest...")

random_forest_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

random_forest_model.fit(X_train, y_train)


# Validation evaluation
print("\nValidation performance:")

evaluate_model(
    "Random Forest - Validation",
    random_forest_model,
    X_validation,
    y_validation
)


# Test evaluation
print("\nTest performance:")

random_forest_results = evaluate_model(
    "Random Forest - Test",
    random_forest_model,
    X_test,
    y_test
)


# ---------------------------------------------------------
# Compare models
# ---------------------------------------------------------

results = pd.DataFrame([
    logistic_results,
    random_forest_results,
])

print("\n" + "=" * 70)
print("FINAL BASELINE COMPARISON")
print("=" * 70)

print(results.to_string(index=False))


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

results.to_csv(OUTPUT_FILE, index=False)

print("\nResults saved to:")
print(OUTPUT_FILE)

print("=" * 70)
