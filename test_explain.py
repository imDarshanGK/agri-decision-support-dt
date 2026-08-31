"""
test_explain.py
-----------------
Covers feature_importance and extract_decision_rules.
"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config
import explain as ex
import data_preprocessing as dp


def test_feature_importance_returns_series_summing_to_one(trained_model, project_dir):
    importances = ex.feature_importance(trained_model, save_plot=True)
    assert isinstance(importances, pd.Series)
    assert set(importances.index) == set(config.FEATURE_COLS)
    assert abs(importances.sum() - 1.0) < 1e-6
    # should be sorted descending
    assert list(importances) == sorted(importances, reverse=True)


def test_feature_importance_saves_plot_file(trained_model, project_dir):
    ex.feature_importance(trained_model, save_plot=True)
    assert os.path.exists("outputs/figures/feature_importance.png")


def test_extract_decision_rules_returns_nonempty_text(trained_model):
    rules = ex.extract_decision_rules(trained_model)
    assert isinstance(rules, str)
    assert len(rules) > 0
    assert "class:" in rules or "|---" in rules  # sklearn export_text format markers


def test_extract_decision_rules_saves_to_file(trained_model, project_dir):
    ex.extract_decision_rules(trained_model, save_path="outputs/decision_rules.txt")
    assert os.path.exists("outputs/decision_rules.txt")
    with open("outputs/decision_rules.txt") as f:
        content = f.read()
    assert len(content) > 0
