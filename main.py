"""
main.py
-------
Runs the full pipeline end-to-end:
Data Preprocessing -> Baseline Training -> Optimization ->
Ranking Demo -> Explainability -> Evaluation

Usage:
    python main.py
"""

import sys
sys.path.insert(0, "src")

from data_preprocessing import run_pipeline as preprocess
from train_baseline import (
    load_processed, train_baseline_model, evaluate_model, save_model
)
from optimize_dt import run_grid_search, cross_validate_model, save_optimized_model
from rank_crops import load_model as load_optimized_model, rank_crops, format_ranking
from explain import feature_importance, extract_decision_rules
from evaluate import run_full_comparison


def main():
    print("\n========== STEP 1: DATA PREPROCESSING ==========")
    preprocess()

    print("\n========== STEP 2: BASELINE DECISION TREE ==========")
    X_train, X_test, y_train, y_test = load_processed()
    baseline = train_baseline_model(X_train, y_train)
    evaluate_model(baseline, X_test, y_test, model_name="Baseline DT")
    save_model(baseline)

    print("\n========== STEP 3: HYPERPARAMETER OPTIMIZATION ==========")
    grid = run_grid_search(X_train, y_train)
    optimized = grid.best_estimator_
    cross_validate_model(optimized, X_train, y_train)
    evaluate_model(optimized, X_test, y_test, model_name="Optimized DT")
    save_optimized_model(optimized)

    print("\n========== STEP 4: CROP RANKING DEMO ==========")
    sample_input = {
        "N": 90, "P": 42, "K": 43, "temperature": 20.8,
        "humidity": 82.0, "ph": 6.5, "rainfall": 202.9,
    }
    results = rank_crops(optimized, sample_input, top_k=3)
    print(format_ranking(results))

    print("\n========== STEP 5: EXPLAINABILITY ==========")
    feature_importance(optimized)
    extract_decision_rules(optimized, save_path="outputs/decision_rules.txt")

    print("\n========== STEP 6: FULL MODEL EVALUATION ==========")
    run_full_comparison()

    print("\n✅ Pipeline complete. Check 'outputs/' and 'models/' for results.")


if __name__ == "__main__":
    main()
