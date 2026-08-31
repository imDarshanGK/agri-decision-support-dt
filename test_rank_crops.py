"""
test_rank_crops.py
--------------------
Covers validate_input and rank_crops, including edge cases and errors.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import rank_crops as rc

VALID_INPUT = {
    "N": 90, "P": 42, "K": 43, "temperature": 20.8,
    "humidity": 82.0, "ph": 6.5, "rainfall": 202.9,
}


def test_validate_input_accepts_valid_values():
    rc.validate_input(VALID_INPUT)  # should not raise


def test_validate_input_rejects_missing_field():
    bad = VALID_INPUT.copy()
    del bad["ph"]
    with pytest.raises(ValueError, match="Missing required field"):
        rc.validate_input(bad)


@pytest.mark.parametrize("field,value", [
    ("N", -5), ("N", 500),
    ("ph", -1), ("ph", 15),
    ("humidity", 150),
    ("rainfall", -10),
    ("temperature", 100),
])
def test_validate_input_rejects_out_of_range(field, value):
    bad = VALID_INPUT.copy()
    bad[field] = value
    with pytest.raises(ValueError, match="out of expected range"):
        rc.validate_input(bad)


def test_rank_crops_returns_top_k_sorted_by_confidence(trained_model):
    results = rc.rank_crops(trained_model, VALID_INPUT, top_k=2)
    assert len(results) == 2
    # descending confidence order
    assert results[0]["confidence"] >= results[1]["confidence"]
    for r in results:
        assert "crop" in r and "confidence" in r
        assert 0.0 <= r["confidence"] <= 1.0


def test_rank_crops_confidences_sum_leq_one(trained_model):
    results = rc.rank_crops(trained_model, VALID_INPUT, top_k=10)
    total = sum(r["confidence"] for r in results)
    assert total <= 1.0001  # floating point tolerance


def test_rank_crops_rejects_model_without_predict_proba():
    class FakeModel:
        pass
    with pytest.raises(AttributeError):
        rc.rank_crops(FakeModel(), VALID_INPUT, top_k=1)


def test_rank_crops_propagates_invalid_input(trained_model):
    bad = VALID_INPUT.copy()
    bad["ph"] = 999
    with pytest.raises(ValueError):
        rc.rank_crops(trained_model, bad, top_k=1)


def test_format_ranking_produces_readable_text():
    results = [{"crop": "rice", "confidence": 0.87}, {"crop": "maize", "confidence": 0.10}]
    text = rc.format_ranking(results)
    assert "1. rice" in text
    assert "87.0%" in text
    assert "2. maize" in text
