# evaluation.py

import numpy as np
from typing import Dict
from src.evaluation.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate predictions using regression metrics.
    
    Returns:
        Dictionary of MSE, MAE, and R².
    """
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2 Score": r2_score(y_true, y_pred)
    }
