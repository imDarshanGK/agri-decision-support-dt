"""
evaluate.py
-----------
Compares Baseline DT, Optimized DT, Random Forest, and KNN on the same
test set. Produces a comparison table and confusion matrix plots.
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

from train_baseline import load_processed

OUTPUT_DIR = "outputs"


def evaluate_and_collect(name, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }


def plot_confusion_matrix(model, X_test, y_test, name: str):
    os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues",
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(f"{OUTPUT_DIR}/figures/confusion_matrix_{safe_name}.png")
    plt.close()
    print(f"[plot_confusion_matrix] Saved for '{name}'")


def run_full_comparison():
    X_train, X_test, y_train, y_test = load_processed()
    results = []

    baseline = joblib.load("models/baseline_dt.pkl")
    results.append(evaluate_and_collect("Baseline DT", baseline, X_test, y_test))
    plot_confusion_matrix(baseline, X_test, y_test, "Baseline DT")

    optimized = joblib.load("models/optimized_dt.pkl")
    results.append(evaluate_and_collect("Optimized DT", optimized, X_test, y_test))
    plot_confusion_matrix(optimized, X_test, y_test, "Optimized DT")

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    results.append(evaluate_and_collect("Random Forest", rf, X_test, y_test))

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    results.append(evaluate_and_collect("KNN", knn, X_test, y_test))

    df_results = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
    print("\n=== Model Comparison ===")
    print(df_results.to_string(index=False))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_results.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)
    print(f"\n[run_full_comparison] Saved to '{OUTPUT_DIR}/model_comparison.csv'")

    return df_results


if __name__ == "__main__":
    run_full_comparison()
