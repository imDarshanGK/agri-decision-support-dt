"""
test_train_baseline.py
------------------------
Covers train_baseline_model, evaluate_model, save_model.
"""

import sys
import os
import joblib
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import train_baseline as tb
import data_preprocessing as dp


def test_train_baseline_model_returns_fitted_tree(synthetic_df):
    X = synthetic_df[dp.FEATURE_COLS]
    y = synthetic_df["label"]
    model = tb.train_baseline_model(X, y)
    assert isinstance(model, DecisionTreeClassifier)
    # a fitted model can predict without raising
    preds = model.predict(X.iloc[:5])
    assert len(preds) == 5


def test_evaluate_model_returns_expected_keys(synthetic_df, trained_model):
    X = synthetic_df[dp.FEATURE_COLS]
    y = synthetic_df["label"]
    metrics = tb.evaluate_model(trained_model, X, y, model_name="Test Model")

    assert metrics["model"] == "Test Model"
    for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_evaluate_model_high_accuracy_on_separable_data(synthetic_df, trained_model):
    # the synthetic fixture is designed to be near-perfectly separable
    X = synthetic_df[dp.FEATURE_COLS]
    y = synthetic_df["label"]
    metrics = tb.evaluate_model(trained_model, X, y)
    assert metrics["accuracy"] > 0.9


def test_save_model_writes_pkl(project_dir, trained_model):
    tb.save_model(trained_model, path="models/test_model.pkl")
    assert os.path.exists("models/test_model.pkl")

    loaded = joblib.load("models/test_model.pkl")
    assert hasattr(loaded, "predict")
