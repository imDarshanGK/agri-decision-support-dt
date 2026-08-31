"""
config.py
---------
Centralized configuration for paths, models, and environment settings.
"""

import os
from pathlib import Path

# Module root (where config.py is located)
_MODULE_ROOT = Path(__file__).parent

# Helper to get current working directory root
def get_project_root():
    """Get project root - uses current working directory dynamically."""
    return Path.cwd()

# Data directories (calculated dynamically based on cwd)
def get_data_dir():
    return get_project_root() / "data"

def get_raw_data_dir():
    return get_data_dir() / "raw"

def get_processed_data_dir():
    return get_data_dir() / "processed"

def get_models_dir():
    return get_project_root() / "models"

def get_outputs_dir():
    return get_project_root() / "outputs"

def get_figures_dir():
    return get_outputs_dir() / "figures"

# Create static references for backward compatibility
# Note: These are evaluated once at import time
DATA_DIR = get_data_dir()
RAW_DATA_DIR = get_raw_data_dir()
PROCESSED_DATA_DIR = get_processed_data_dir()
MODELS_DIR = get_models_dir()
OUTPUTS_DIR = get_outputs_dir()
FIGURES_DIR = get_figures_dir()

# Data files
RAW_DATASET_PATH = RAW_DATA_DIR / "Crop_recommendation.csv"
# Fallback location in original module location
DATASET_ALT_PATH = _MODULE_ROOT / "Crop_recommendation.csv"

# Model files
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_dt.pkl"
OPTIMIZED_MODEL_PATH = MODELS_DIR / "optimized_dt.pkl"
OPTIMIZED_MODEL_ALT_PATH = _MODULE_ROOT / "optimized_dt.pkl"  # Fallback location in module root

# Feature configuration
FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COL = "label"

# Model hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Grid search configuration
GRID_SEARCH_SCORING = "f1_macro"
GRID_SEARCH_N_JOBS = -1

# Input validation ranges
INPUT_RANGES = {
    "N": (0, 200),
    "P": (0, 200),
    "K": (0, 200),
    "temperature": (-10, 60),
    "humidity": (0, 100),
    "ph": (0, 14),
    "rainfall": (0, 500),
}

# API configuration
API_TITLE = "Crop Recommendation Decision Support API"
API_VERSION = "1.0.0"
API_CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def ensure_dirs():
    """Create required directories if they don't exist."""
    dirs = [
        get_models_dir(),
        get_processed_data_dir(),
        get_figures_dir(),
        get_outputs_dir()
    ]
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def resolve_data_path():
    """Resolve the correct path to the raw dataset."""
    raw_path = get_raw_data_dir() / "Crop_recommendation.csv"
    if raw_path.exists():
        return raw_path
    if DATASET_ALT_PATH.exists():
        # Copy it to the expected location
        raw_data_dir = get_raw_data_dir()
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        if not raw_path.exists():
            import shutil
            shutil.copy2(DATASET_ALT_PATH, raw_path)
        return raw_path
    # Return the expected path (will raise an error if it doesn't exist)
    return raw_path


def resolve_model_path(model_type: str = "optimized"):
    """Resolve the correct path to a trained model.
    
    Checks CWD first, then falls back to repo root if running from the repo directory.
    """
    cwd = get_project_root()
    
    # Check CWD first
    if model_type == "optimized":
        cwd_candidates = [
            cwd / "models" / "optimized_dt.pkl",
            cwd / "optimized_dt.pkl",
        ]
        for candidate in cwd_candidates:
            if candidate.exists():
                return candidate
        
        # Only check repo root if we're not already in the repo
        if cwd != _MODULE_ROOT:
            if (_MODULE_ROOT / "models" / "optimized_dt.pkl").exists():
                return _MODULE_ROOT / "models" / "optimized_dt.pkl"
            if (_MODULE_ROOT / "optimized_dt.pkl").exists():
                return _MODULE_ROOT / "optimized_dt.pkl"
        
        # Return CWD path as default (for error messages)
        return cwd / "models" / "optimized_dt.pkl"
        
    elif model_type == "baseline":
        cwd_candidate = cwd / "models" / "baseline_dt.pkl"
        if cwd_candidate.exists():
            return cwd_candidate
        if cwd != _MODULE_ROOT:
            if (_MODULE_ROOT / "models" / "baseline_dt.pkl").exists():
                return _MODULE_ROOT / "models" / "baseline_dt.pkl"
        return cwd_candidate
        
    raise ValueError(f"Unknown model type: {model_type}")
