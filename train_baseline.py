"""
train_baseline.py
------------------
Trains a default (unoptimized) Decision Tree classifier as a baseline.
"""

import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"


def load_processed(dir_path: str = PROCESSED_DIR):
    X_train = pd.read_csv(f"{dir_path}/X_train.csv")
    X_test = pd.read_csv(f"{dir_path}/X_test.csv")
    y_train = pd.read_csv(f"{dir_path}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{dir_path}/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def train_baseline_model(X_train, y_train, random_state: int = 42) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name: str = "Baseline DT") -> dict:
    y_pred = model.predict(X_test)
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }
    print(f"\n=== {model_name} Results ===")
    for k, v in metrics.items():
        if k != "model":
            print(f"{k}: {v:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return metrics


def save_model(model, path: str = f"{MODEL_DIR}/baseline_dt.pkl"):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, path)
    print(f"[save_model] Saved baseline model to '{path}'")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_processed()
    model = train_baseline_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)
