"""api.py
------
FastAPI backend for the Crop Recommendation Decision Support System.
Serves the optimized Decision Tree model and the static frontend.

Run with:
    uvicorn api:app --reload
Then open http://127.0.0.1:8000
"""

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import config
from rank_crops import rank_crops, validate_input
from explain import extract_decision_rules
from utils import load_model, is_model_available, get_logger, ModelNotFoundError

logger = get_logger(__name__)

BASE_DIR = Path(__file__).parent
ROOT_HTML = BASE_DIR / "index.html"

app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description="Crop recommendation system with hyperparameter-optimized Decision Trees",
)

# Configure CORS properly - only allow specified origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

_model_cache: Any = None


def get_model_cached() -> Any:
    """Get or load the optimized decision tree model.
    
    Returns:
        The loaded scikit-learn DecisionTreeClassifier model
        
    Raises:
        HTTPException: If the model cannot be loaded (503 Service Unavailable)
    """
    global _model_cache
    if _model_cache is None:
        try:
            _model_cache = load_model("optimized")
        except ModelNotFoundError as e:
            logger.error(f"Model loading failed: {e}")
            raise HTTPException(
                status_code=503,
                detail="Model not found. Run 'python main.py' first to train it.",
            ) from e
    return _model_cache


class CropRequest(BaseModel):
    """Request model for crop prediction."""
    N: float = Field(..., ge=0, le=200, description="Nitrogen content (0-200)")
    P: float = Field(..., ge=0, le=200, description="Phosphorus content (0-200)")
    K: float = Field(..., ge=0, le=200, description="Potassium content (0-200)")
    temperature: float = Field(..., ge=-10, le=60, description="Temperature in Celsius (-10 to 60)")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity % (0-100)")
    ph: float = Field(..., ge=0, le=14, description="Soil pH (0-14)")
    rainfall: float = Field(..., ge=0, le=500, description="Rainfall in mm (0-500)")
    top_k: int = Field(default=3, ge=1, le=5, description="Number of top recommendations")


class CropResult(BaseModel):
    """A single crop recommendation with confidence score."""
    crop: str
    confidence: float


class CropResponse(BaseModel):
    """Response model for crop prediction results."""
    recommendations: list[CropResult]
    top_features: dict[str, float]


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Health check endpoint.
    
    Returns:
        Status and model availability information
    """
    return {
        "status": "ok",
        "version": config.API_VERSION,
        "model_loaded": is_model_available("optimized"),
    }


@app.post("/api/predict", response_model=CropResponse)
def predict(req: CropRequest) -> CropResponse:
    """Predict ranked crop recommendations based on soil and climate parameters.
    
    Args:
        req: Crop request with soil/climate readings
        
    Returns:
        Ranked crop recommendations with confidence scores and feature importance
        
    Raises:
        HTTPException: If input validation fails or model is unavailable
    """
    try:
        model = get_model_cached()
    except HTTPException:
        raise
    
    input_dict = req.model_dump(exclude={"top_k"})

    try:
        validate_input(input_dict)
    except ValueError as e:
        logger.warning(f"Input validation failed: {e}")
        raise HTTPException(status_code=422, detail=str(e)) from e

    try:
        results = rank_crops(model, input_dict, top_k=req.top_k)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed") from e

    importances = dict(zip(config.FEATURE_COLS, model.feature_importances_.tolist()))

    return CropResponse(
        recommendations=[CropResult(**r) for r in results],
        top_features=importances,
    )


@app.get("/api/rules")
def get_rules() -> dict[str, str]:
    """Get the decision tree rules in human-readable format.
    
    Returns:
        Decision tree rules as text (truncated to 5000 chars)
        
    Raises:
        HTTPException: If model is unavailable
    """
    try:
        model = get_model_cached()
    except HTTPException:
        raise
    
    try:
        rules = extract_decision_rules(model)
        return {"rules": rules[:5000]}
    except Exception as e:
        logger.error(f"Failed to extract rules: {e}")
        raise HTTPException(status_code=500, detail="Failed to extract rules") from e


# Serve the frontend
@app.get("/")
def serve_root() -> FileResponse:
    """Serve the main UI."""
    if ROOT_HTML.exists():
        return FileResponse(ROOT_HTML)
    raise HTTPException(status_code=404, detail="Frontend not found")


@app.get("/index.html")
def serve_index_html() -> FileResponse:
    """Serve the index HTML file."""
    if ROOT_HTML.exists():
        return FileResponse(ROOT_HTML)
    raise HTTPException(status_code=404, detail="Frontend not found")


frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
