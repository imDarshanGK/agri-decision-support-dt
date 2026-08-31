"""
rank_crops.py
-------------
Returns the top-k most suitable crops with confidence scores,
using predict_proba, instead of a single prediction.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd

import config
from utils import load_model as utils_load_model, get_logger

logger = get_logger(__name__)


def validate_input(input_dict: Dict[str, float]) -> None:
    """Validate crop recommendation input parameters.
    
    Args:
        input_dict: Dictionary with field readings
        
    Raises:
        ValueError: If any field is missing or out of valid range
    """
    for field, (lo, hi) in config.INPUT_RANGES.items():
        val = input_dict.get(field)
        if val is None:
            raise ValueError(f"Missing required field: {field}")
        if not isinstance(val, (int, float)):
            raise ValueError(f"'{field}' must be numeric, got {type(val).__name__}")
        if not (lo <= val <= hi):
            raise ValueError(f"'{field}'={val} is out of expected range [{lo}, {hi}]")


def rank_crops(model: Any, input_dict: Dict[str, float], top_k: int = 3) -> List[Dict[str, Any]]:
    """Rank crops by suitability given soil and climate parameters.
    
    Args:
        model: Trained decision tree classifier
        input_dict: Dictionary with field readings
        top_k: Number of top recommendations to return
        
    Returns:
        List of ranked crops with confidence scores
        
    Raises:
        ValueError: If input validation fails
        AttributeError: If model doesn't support predict_proba
    """
    validate_input(input_dict)
    X = pd.DataFrame([input_dict])[config.FEATURE_COLS]

    if not hasattr(model, "predict_proba"):
        raise AttributeError("Model does not support predict_proba.")

    probs = model.predict_proba(X)[0]
    classes = model.classes_

    ranked_idx = np.argsort(probs)[::-1][:top_k]
    results = [
        {"crop": classes[i], "confidence": round(float(probs[i]), 4)}
        for i in ranked_idx
    ]
    logger.debug(f"Ranked top {top_k} crops for input: {results}")
    return results


def format_ranking(results: List[Dict[str, Any]]) -> str:
    """Format ranking results as human-readable text.
    
    Args:
        results: List of ranked crop results
        
    Returns:
        Formatted text representation
    """
    return "\n".join(
        f"{i+1}. {r['crop']} (confidence: {r['confidence']*100:.1f}%)"
        for i, r in enumerate(results)
    )


if __name__ == "__main__":
    try:
        model = utils_load_model("optimized")
        sample_input = {
            "N": 90, "P": 42, "K": 43, "temperature": 20.8,
            "humidity": 82.0, "ph": 6.5, "rainfall": 202.9,
        }
        top_crops = rank_crops(model, sample_input, top_k=3)
        print("Top crop recommendations:")
        print(format_ranking(top_crops))
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
