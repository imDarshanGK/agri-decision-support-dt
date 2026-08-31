"""
data_preprocessing.py
----------------------
Loads the Crop Recommendation dataset, cleans it, and produces a
train/test split saved to data/processed/.

Expected raw file: data/raw/Crop_recommendation.csv
Expected columns: N, P, K, temperature, humidity, ph, rainfall, label
"""

import os
import shutil
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

import config
from utils import get_logger, DataNotFoundError

logger = get_logger(__name__)


def load_data(path: str = None) -> pd.DataFrame:
    """Load the crop recommendation dataset.
    
    Args:
        path: Optional custom path to dataset
        
    Returns:
        Loaded dataframe
        
    Raises:
        DataNotFoundError: If dataset cannot be found
    """
    if path is None:
        path = str(config.resolve_data_path())
    
    if not os.path.exists(path):
        raise DataNotFoundError(
            f"Dataset not found at '{path}'. Download the Kaggle "
            "'Crop Recommendation Dataset' and place it there."
        )
    
    logger.info(f"Loading dataset from {path}")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the dataset.
    
    Args:
        df: Raw dataframe
        
    Returns:
        Cleaned dataframe
    """
    missing_cols = [c for c in config.FEATURE_COLS + [config.TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=config.FEATURE_COLS + [config.TARGET_COL])
    logger.info(f"[clean_data] Rows before: {before} -> after cleaning: {len(df)}")

    df = df[(df["ph"] >= 0) & (df["ph"] <= 14)]
    df = df[df["humidity"] >= 0]
    df = df[df["rainfall"] >= 0]

    return df.reset_index(drop=True)


def split_data(df: pd.DataFrame, test_size: float = None, random_state: int = None) -> Tuple:
    """Split dataset into train and test sets.
    
    Args:
        df: Input dataframe
        test_size: Test set fraction (default from config)
        random_state: Random seed (default from config)
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if test_size is None:
        test_size = config.TEST_SIZE
    if random_state is None:
        random_state = config.RANDOM_STATE
        
    X = df[config.FEATURE_COLS]
    y = df[config.TARGET_COL]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def save_processed(X_train, X_test, y_train, y_test, out_dir: str = None) -> None:
    """Save processed data splits to CSV files.
    
    Args:
        X_train, X_test, y_train, y_test: Data splits
        out_dir: Output directory (default from config)
    """
    if out_dir is None:
        out_dir = str(config.get_processed_data_dir())
        
    os.makedirs(out_dir, exist_ok=True)
    X_train.to_csv(f"{out_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{out_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{out_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{out_dir}/y_test.csv", index=False)
    logger.info(f"[save_processed] Saved processed splits to '{out_dir}'/") 


def run_pipeline() -> Tuple:
    """Run the full data preprocessing pipeline.
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    config.ensure_dirs()
    df = load_data()
    df = clean_data(df)
    logger.info(f"[run_pipeline] Class distribution:\n{df[config.TARGET_COL].value_counts()}")
    X_train, X_test, y_train, y_test = split_data(df)
    save_processed(X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_pipeline()

