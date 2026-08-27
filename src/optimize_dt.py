"""
optimize_dt.py
---------------
Hyperparameter-tunes a Decision Tree using GridSearchCV, then evaluates
and saves the optimized model.
"""

import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

from train_baseline import load_processed, evaluate_model, MODEL_DIR

PARAM_GRID = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 5, 7, 10, 15, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "ccp_alpha": [0.0, 0.001, 0.01],
}


def run_grid_search(X_train, y_train, cv: int = 5, random_state: int = 42) -> GridSearchCV:
    base_model = DecisionTreeClassifier(random_state=random_state)
    grid = GridSearchCV(
        estimator=base_model,
        param_grid=PARAM_GRID,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    print(f"[run_grid_search] Best params: {grid.best_params_}")
    print(f"[run_grid_search] Best CV F1 (macro): {grid.best_score_:.4f}")
    return grid


def cross_validate_model(model, X_train, y_train, cv: int = 5):
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
    print(f"[cross_validate_model] F1 (macro) per fold: {scores}")
    print(f"[cross_validate_model] Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")
    return scores


def save_optimized_model(model, path: str = f"{MODEL_DIR}/optimized_dt.pkl"):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, path)
    print(f"[save_optimized_model] Saved optimized model to '{path}'")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_processed()

    grid = run_grid_search(X_train, y_train)
    best_model = grid.best_estimator_

    cross_validate_model(best_model, X_train, y_train)
    evaluate_model(best_model, X_test, y_test, model_name="Optimized DT")

    save_optimized_model(best_model)

    pd.Series(grid.best_params_).to_csv(f"{MODEL_DIR}/best_params.csv")
    print(f"[main] Best params saved to '{MODEL_DIR}/best_params.csv'")
