"""
test_api.py
------------
Covers the FastAPI endpoints using TestClient against a real trained
model, so requests exercise the actual prediction path end-to-end.
"""

import sys
import os
import joblib
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
import data_preprocessing as dp

VALID_PAYLOAD = {
    "N": 90, "P": 42, "K": 43, "temperature": 20.8,
    "humidity": 82.0, "ph": 6.5, "rainfall": 202.9, "top_k": 2,
}


def _make_client(project_dir, synthetic_df, trained_model):
    os.makedirs("models", exist_ok=True)
    joblib.dump(trained_model, "models/optimized_dt.pkl")

    import api as api_module
    importlib.reload(api_module)  # reset the module-level _model cache
    return TestClient(api_module.app)


def test_health_endpoint_reports_model_loaded(project_dir, synthetic_df, trained_model):
    client = _make_client(project_dir, synthetic_df, trained_model)
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    assert res.json()["model_loaded"] is True


def test_health_endpoint_reports_no_model(project_dir):
    import api as api_module
    importlib.reload(api_module)
    client = TestClient(api_module.app)
    res = client.get("/api/health")
    assert res.json()["model_loaded"] is False


def test_predict_returns_ranked_crops(project_dir, synthetic_df, trained_model):
    client = _make_client(project_dir, synthetic_df, trained_model)
    res = client.post("/api/predict", json=VALID_PAYLOAD)
    assert res.status_code == 200
    body = res.json()
    assert len(body["recommendations"]) == 2
    assert body["recommendations"][0]["confidence"] >= body["recommendations"][1]["confidence"]
    assert set(body["top_features"].keys()) == set(dp.FEATURE_COLS)


def test_predict_rejects_out_of_range_input(project_dir, synthetic_df, trained_model):
    client = _make_client(project_dir, synthetic_df, trained_model)
    bad_payload = VALID_PAYLOAD.copy()
    bad_payload["ph"] = 999
    res = client.post("/api/predict", json=bad_payload)
    assert res.status_code == 422


def test_predict_missing_model_returns_503(project_dir):
    import api as api_module
    importlib.reload(api_module)
    client = TestClient(api_module.app)
    res = client.post("/api/predict", json=VALID_PAYLOAD)
    assert res.status_code == 503


def test_rules_endpoint_returns_text(project_dir, synthetic_df, trained_model):
    client = _make_client(project_dir, synthetic_df, trained_model)
    res = client.get("/api/rules")
    assert res.status_code == 200
    assert len(res.json()["rules"]) > 0
