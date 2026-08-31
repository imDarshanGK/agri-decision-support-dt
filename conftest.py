"""
conftest.py
-----------
Shared pytest fixtures: builds a small synthetic dataset and a trained
model in a temp directory so tests don't depend on the real Kaggle CSV
or mutate the real project's data/models folders.
"""

import os
import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

import config


@pytest.fixture(scope="session")
def synthetic_df():
    """A small, fast synthetic dataset with a clear separable pattern."""
    rng = np.random.RandomState(0)
    crops = ["rice", "maize", "cotton", "coffee"]
    rows = []
    for i, crop in enumerate(crops):
        base = np.array([20, 20, 20, 15, 20, 1, 40]) * (i + 1)
        for _ in range(40):
            noise = rng.normal(0, 2, size=7)
            vals = base + noise
            rows.append({
                "N": max(0, vals[0]), "P": max(0, vals[1]), "K": max(0, vals[2]),
                "temperature": np.clip(vals[3], -10, 60),
                "humidity": np.clip(vals[4], 0, 100),
                "ph": np.clip(vals[5], 0, 14),
                "rainfall": np.clip(vals[6], 0, 500),
                "label": crop,
            })
    return pd.DataFrame(rows)


@pytest.fixture()
def project_dir(tmp_path, synthetic_df, monkeypatch):
    """
    Creates an isolated project directory (data/, models/, outputs/)
    populated with the synthetic dataset, and chdirs into it so every
    module's relative paths ('data/raw/...', 'models/...') resolve here.
    """
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "outputs" / "figures").mkdir(parents=True)

    synthetic_df.to_csv(tmp_path / "data" / "raw" / "Crop_recommendation.csv", index=False)

    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture()
def trained_model(synthetic_df):
    """A quickly-trained model for tests that don't need the full pipeline."""
    X = synthetic_df[config.FEATURE_COLS]
    y = synthetic_df["label"]
    model = DecisionTreeClassifier(random_state=config.RANDOM_STATE, max_depth=5)
    model.fit(X, y)
    return model
