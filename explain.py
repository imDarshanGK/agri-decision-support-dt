"""
explain.py
----------
Explainability utilities: feature importance, SHAP analysis,
and human-readable decision rule extraction.
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import export_text

MODEL_PATH = "models/optimized_dt.pkl"
OUTPUT_DIR = "outputs"
FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def load_model(path: str = None):
    if path is None:
        candidates = ["models/optimized_dt.pkl", "optimized_dt.pkl"]
        for candidate in candidates:
            if os.path.exists(candidate):
                path = candidate
                break
        if path is None:
            path = MODEL_PATH
    return joblib.load(path)


def feature_importance(model, feature_names=FEATURE_COLS, save_plot: bool = True) -> pd.Series:
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)
    print("Feature importances:\n", importances)

    if save_plot:
        os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)
        plt.figure(figsize=(8, 5))
        importances.plot(kind="bar")
        plt.title("Decision Tree Feature Importance")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/figures/feature_importance.png")
        plt.close()
        print(f"[feature_importance] Plot saved to '{OUTPUT_DIR}/figures/feature_importance.png'")

    return importances


def extract_decision_rules(model, feature_names=FEATURE_COLS, save_path: str = None) -> str:
    rules_text = export_text(model, feature_names=feature_names)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write(rules_text)
        print(f"[extract_decision_rules] Rules saved to '{save_path}'")
    return rules_text


def run_shap_analysis(model, X_sample: pd.DataFrame, save_plot: bool = True):
    """Requires: pip install shap"""
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if save_plot:
        os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/figures/shap_summary.png")
        plt.close()
        print(f"[run_shap_analysis] Saved to '{OUTPUT_DIR}/figures/shap_summary.png'")

    return shap_values


if __name__ == "__main__":
    model = load_model()
    feature_importance(model)
    extract_decision_rules(model, save_path=f"{OUTPUT_DIR}/decision_rules.txt")

    try:
        X_test = pd.read_csv("data/processed/X_test.csv")
        run_shap_analysis(model, X_test.sample(min(100, len(X_test)), random_state=42))
    except FileNotFoundError:
        print("[main] Skipping SHAP: processed test data not found.")
