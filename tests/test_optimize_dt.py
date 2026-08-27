"""
test_optimize_dt.py
---------------------
Covers run_grid_search, cross_validate_model, save_optimized_model.
Uses a tiny param grid override to keep tests fast.
"""

import sys
import os
import joblib
from sklearn.model_selection import GridSearchCV

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import optimize_dt as opt
import data_preprocessing as dp


def test_run_grid_search_returns_fitted_grid(synthetic_df, monkeypatch):
    # shrink the grid so the test runs fast
    monkeypatch.setattr(opt, "PARAM_GRID", {
        "criterion": ["gini"],
        "max_depth": [3, 5],
        "min_samples_split": [2],
        "min_samples_leaf": [1],
        "ccp_alpha": [0.0],
    })
    X = synthetic_df[dp.FEATURE_COLS]
    y = synthetic_df["label"]

    grid = opt.run_grid_search(X, y, cv=3)
    assert isinstance(grid, GridSearchCV)
    assert grid.best_estimator_ is not None
    assert 0.0 <= grid.best_score_ <= 1.0


def test_cross_validate_model_returns_scores(synthetic_df, trained_model):
    X = synthetic_df[dp.FEATURE_COLS]
    y = synthetic_df["label"]
    scores = opt.cross_validate_model(trained_model, X, y, cv=3)
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_save_optimized_model_writes_pkl(project_dir, trained_model):
    opt.save_optimized_model(trained_model, path="models/optimized_dt.pkl")
    assert os.path.exists("models/optimized_dt.pkl")
    loaded = joblib.load("models/optimized_dt.pkl")
    assert hasattr(loaded, "predict_proba")
