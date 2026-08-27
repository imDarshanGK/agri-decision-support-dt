"""
rank_crops.py
-------------
Returns the top-k most suitable crops with confidence scores,
using predict_proba, instead of a single prediction.
"""

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "models/optimized_dt.pkl"
FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def load_model(path: str = MODEL_PATH):
    return joblib.load(path)


def validate_input(input_dict: dict):
    ranges = {
        "N": (0, 200), "P": (0, 200), "K": (0, 200),
        "temperature": (-10, 60), "humidity": (0, 100),
        "ph": (0, 14), "rainfall": (0, 500),
    }
    for field, (lo, hi) in ranges.items():
        val = input_dict.get(field)
        if val is None:
            raise ValueError(f"Missing required field: {field}")
        if not (lo <= val <= hi):
            raise ValueError(f"'{field}'={val} is out of expected range [{lo}, {hi}]")


def rank_crops(model, input_dict: dict, top_k: int = 3) -> list:
    validate_input(input_dict)
    X = pd.DataFrame([input_dict])[FEATURE_COLS]

    if not hasattr(model, "predict_proba"):
        raise AttributeError("Model does not support predict_proba.")

    probs = model.predict_proba(X)[0]
    classes = model.classes_

    ranked_idx = np.argsort(probs)[::-1][:top_k]
    results = [
        {"crop": classes[i], "confidence": round(float(probs[i]), 4)}
        for i in ranked_idx
    ]
    return results


def format_ranking(results: list) -> str:
    return "\n".join(
        f"{i+1}. {r['crop']} (confidence: {r['confidence']*100:.1f}%)"
        for i, r in enumerate(results)
    )


if __name__ == "__main__":
    model = load_model()
    sample_input = {
        "N": 90, "P": 42, "K": 43, "temperature": 20.8,
        "humidity": 82.0, "ph": 6.5, "rainfall": 202.9,
    }
    top_crops = rank_crops(model, sample_input, top_k=3)
    print("Top crop recommendations:")
    print(format_ranking(top_crops))
