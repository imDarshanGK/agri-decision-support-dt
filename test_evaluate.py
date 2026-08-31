"""
test_evaluate.py
------------------
Covers evaluate_and_collect, plot_confusion_matrix, run_full_comparison.
"""

import sys
import os
import joblib
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config
import evaluate as ev
import data_preprocessing as dp


def test_evaluate_and_collect_returns_metric_dict(synthetic_df, trained_model):
    X = synthetic_df[config.FEATURE_COLS]
    y = synthetic_df["label"]
    result = ev.evaluate_and_collect("Test Model", trained_model, X, y)
    assert result["model"] == "Test Model"
    assert all(0.0 <= result[k] <= 1.0 for k in
               ["accuracy", "precision_macro", "recall_macro", "f1_macro"])


def test_plot_confusion_matrix_saves_file(project_dir, synthetic_df, trained_model):
    X = synthetic_df[config.FEATURE_COLS]
    y = synthetic_df["label"]
    ev.plot_confusion_matrix(trained_model, X, y, "Test Model")
    assert os.path.exists("outputs/figures/confusion_matrix_test_model.png")


def test_run_full_comparison_end_to_end(project_dir, synthetic_df, trained_model):
    # set up processed data + saved models the way the real pipeline would
    X_train, X_test, y_train, y_test = dp.split_data(synthetic_df, test_size=0.3)
    dp.save_processed(X_train, X_test, y_train, y_test, out_dir="data/processed")

    os.makedirs("models", exist_ok=True)
    joblib.dump(trained_model, "models/baseline_dt.pkl")
    joblib.dump(trained_model, "models/optimized_dt.pkl")

    df_results = ev.run_full_comparison()

    assert isinstance(df_results, pd.DataFrame)
    assert set(df_results["model"]) == {"Baseline DT", "Optimized DT", "Random Forest", "KNN"}
    assert os.path.exists("outputs/model_comparison.csv")
    # results should be sorted by f1_macro descending
    assert list(df_results["f1_macro"]) == sorted(df_results["f1_macro"], reverse=True)
