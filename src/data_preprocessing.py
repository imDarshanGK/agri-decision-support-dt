"""
data_preprocessing.py
----------------------
Loads the Crop Recommendation dataset, cleans it, and produces a
train/test split saved to data/processed/.

Expected raw file: data/raw/Crop_recommendation.csv
Expected columns: N, P, K, temperature, humidity, ph, rainfall, label
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/Crop_recommendation.csv"
PROCESSED_DIR = "data/processed"
FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COL = "label"


def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Download the Kaggle "
            "'Crop Recommendation Dataset' and place it there."
        )
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    print(f"[clean_data] Rows before: {before} -> after cleaning: {len(df)}")

    df = df[(df["ph"] >= 0) & (df["ph"] <= 14)]
    df = df[df["humidity"] >= 0]
    df = df[df["rainfall"] >= 0]

    return df.reset_index(drop=True)


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def save_processed(X_train, X_test, y_train, y_test, out_dir: str = PROCESSED_DIR):
    os.makedirs(out_dir, exist_ok=True)
    X_train.to_csv(f"{out_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{out_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{out_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{out_dir}/y_test.csv", index=False)
    print(f"[save_processed] Saved processed splits to '{out_dir}/'")


def run_pipeline():
    df = load_data()
    df = clean_data(df)
    print("[run_pipeline] Class distribution:\n", df[TARGET_COL].value_counts())
    X_train, X_test, y_train, y_test = split_data(df)
    save_processed(X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_pipeline()
