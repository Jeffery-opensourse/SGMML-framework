"""Evaluation metrics (Section 2.5).

Five standard regression metrics are used throughout the paper:
coefficient of determination (R2), mean square error (MSE), root mean
square error (RMSE), mean absolute error (MAE) and mean absolute
percentage error (MAPE).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute R2, MSE, RMSE, MAE and MAPE for a prediction."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def segmented_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Dict]:
    """Overall metrics plus errors split into low / mid / high strength.

    The manuscript discusses a systematic underestimation of high-strength
    samples; this breakdown makes that behaviour explicit. Percentile
    boundaries follow the 33.3 / 66.7 quantiles of the true values.
    """
    result = {"overall": calc_metrics(y_true, y_pred)}
    q1, q2 = np.percentile(y_true, [33.3, 66.7])
    segments = {
        "low (<33.3%)": y_true < q1,
        "mid (33.3-66.7%)": (y_true >= q1) & (y_true < q2),
        "high (>=66.7%)": y_true >= q2,
    }
    for name, mask in segments.items():
        if mask.sum() == 0:
            continue
        sub = calc_metrics(y_true[mask], y_pred[mask])
        sub["n"] = int(mask.sum())
        sub["bias"] = float(np.mean(y_pred[mask] - y_true[mask]))
        result[name] = sub
    return result
