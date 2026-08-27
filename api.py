"""
api.py
------
FastAPI backend for the Crop Recommendation Decision Support System.
Serves the optimized Decision Tree model and the static frontend.

Run with:
    uvicorn api:app --reload
Then open http://127.0.0.1:8000
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rank_crops import rank_crops, validate_input, FEATURE_COLS
from explain import extract_decision_rules

MODEL_PATH = "models/optimized_dt.pkl"

app = FastAPI(title="Crop Recommendation Decision Support API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail="Model not found. Run 'python main.py' first to train it.",
            )
        _model = joblib.load(MODEL_PATH)
    return _model


class CropRequest(BaseModel):
    N: float = Field(..., ge=0, le=200, description="Nitrogen content")
    P: float = Field(..., ge=0, le=200, description="Phosphorus content")
    K: float = Field(..., ge=0, le=200, description="Potassium content")
    temperature: float = Field(..., ge=-10, le=60, description="Temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity %")
    ph: float = Field(..., ge=0, le=14, description="Soil pH")
    rainfall: float = Field(..., ge=0, le=500, description="Rainfall in mm")
    top_k: int = Field(3, ge=1, le=5, description="Number of recommendations")


class CropResult(BaseModel):
    crop: str
    confidence: float


class CropResponse(BaseModel):
    recommendations: list[CropResult]
    top_features: dict


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)}


@app.post("/api/predict", response_model=CropResponse)
def predict(req: CropRequest):
    model = get_model()
    input_dict = req.model_dump(exclude={"top_k"})

    try:
        validate_input(input_dict)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    results = rank_crops(model, input_dict, top_k=req.top_k)

    importances = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))

    return CropResponse(
        recommendations=[CropResult(**r) for r in results],
        top_features=importances,
    )


@app.get("/api/rules")
def get_rules():
    model = get_model()
    rules = extract_decision_rules(model)
    return {"rules": rules[:5000]}


# Serve the frontend
if os.path.isdir("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
