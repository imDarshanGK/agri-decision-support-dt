"""
test_data_preprocessing.py
---------------------------
Covers load_data, clean_data, split_data, save_processed, run_pipeline.
"""

import sys
import os
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data_preprocessing as dp


def test_load_data_missing_file_raises(project_dir):
    os.remove("data/raw/Crop_recommendation.csv")
    with pytest.raises(FileNotFoundError):
        dp.load_data()


def test_load_data_reads_csv(project_dir):
    df = dp.load_data()
    assert not df.empty
    assert "label" in df.columns


def test_clean_data_missing_column_raises(synthetic_df):
    bad_df = synthetic_df.drop(columns=["ph"])
    with pytest.raises(ValueError):
        dp.clean_data(bad_df)


def test_clean_data_removes_duplicates(synthetic_df):
    dup_df = pd.concat([synthetic_df, synthetic_df.iloc[:5]], ignore_index=True)
    cleaned = dp.clean_data(dup_df)
    assert len(cleaned) == len(dup_df.drop_duplicates())


def test_clean_data_filters_invalid_ph(synthetic_df):
    bad_df = synthetic_df.copy()
    bad_df.loc[0, "ph"] = 20  # invalid, > 14
    cleaned = dp.clean_data(bad_df)
    assert (cleaned["ph"] <= 14).all()
    assert (cleaned["ph"] >= 0).all()


def test_split_data_shapes(synthetic_df):
    X_train, X_test, y_train, y_test = dp.split_data(synthetic_df, test_size=0.25)
    assert len(X_train) + len(X_test) == len(synthetic_df)
    assert set(X_train.columns) == set(dp.FEATURE_COLS)
    assert len(y_train) == len(X_train)


def test_split_data_is_stratified(synthetic_df):
    X_train, X_test, y_train, y_test = dp.split_data(synthetic_df, test_size=0.5)
    # every class in the original data should still appear in both splits
    assert set(y_train.unique()) == set(synthetic_df["label"].unique())
    assert set(y_test.unique()) == set(synthetic_df["label"].unique())


def test_save_processed_writes_four_files(project_dir, synthetic_df):
    X_train, X_test, y_train, y_test = dp.split_data(synthetic_df)
    dp.save_processed(X_train, X_test, y_train, y_test, out_dir="data/processed")

    for fname in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
        assert os.path.exists(f"data/processed/{fname}")


def test_run_pipeline_end_to_end(project_dir):
    X_train, X_test, y_train, y_test = dp.run_pipeline()
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert os.path.exists("data/processed/X_train.csv")
    # no missing values should remain
    assert not X_train.isna().any().any()
