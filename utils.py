"""
utils.py
--------
Shared utilities: logging, model loading, error handling.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Any

import joblib

from config import (
    LOG_LEVEL,
    LOG_FORMAT,
    resolve_model_path,
    OPTIMIZED_MODEL_PATH,
    BASELINE_MODEL_PATH,
)

# Configure logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    """Raised when a required trained model is not found."""
    pass


class DataNotFoundError(Exception):
    """Raised when required data files are not found."""
    pass


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a given module name."""
    return logging.getLogger(name)


def load_model(model_type: str = "optimized") -> Any:
    """
    Load a trained model from disk.
    
    Args:
        model_type: Either "optimized" or "baseline"
        
    Returns:
        The loaded scikit-learn model
        
    Raises:
        ModelNotFoundError: If the model file is not found
    """
    from pathlib import Path as PathlibPath
    
    path = resolve_model_path(model_type)
    
    if not PathlibPath(path).exists():
        raise ModelNotFoundError(
            f"Model not found at {path}. Run 'python main.py' to train the model."
        )
    
    try:
        model = joblib.load(path)
        get_logger(__name__).info(f"Loaded {model_type} model from {path}")
        return model
    except Exception as e:
        raise ModelNotFoundError(f"Failed to load model from {path}: {e}") from e


def save_model(model: Any, model_type: str = "optimized") -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: The scikit-learn model to save
        model_type: Either "optimized" or "baseline"
    """
    if model_type == "optimized":
        path = OPTIMIZED_MODEL_PATH
    elif model_type == "baseline":
        path = BASELINE_MODEL_PATH
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(model, path)
        get_logger(__name__).info(f"Saved {model_type} model to {path}")
    except Exception as e:
        get_logger(__name__).error(f"Failed to save model to {path}: {e}")
        raise


def is_model_available(model_type: str = "optimized") -> bool:
    """Check if a trained model is available."""
    path = resolve_model_path(model_type)
    return Path(path).exists()
