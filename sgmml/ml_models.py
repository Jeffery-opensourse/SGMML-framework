"""Conventional machine learning baselines (Section 2.3.2).

Eight classical models are benchmarked on the structured numerical
descriptors: linear regression (LR), Ridge, Lasso, k-nearest neighbors
(KNN), support vector regression (SVR), random forest (RF), gradient
boosting regression (GBR) and XGBoost (XGB). Each model is tuned with
grid search and 5-fold cross-validation on the training set.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

from .config import CV_FOLDS, RANDOM_SEED
from .metrics import calc_metrics

# Model name -> (estimator, parameter grid). Grids are kept small so the
# whole benchmark finishes quickly on a modest workstation.
MODEL_GRIDS: Dict[str, tuple] = {
    "LR": (LinearRegression(), {}),
    "Ridge": (Ridge(), {"alpha": [0.1, 1.0, 10.0, 100.0]}),
    "Lasso": (Lasso(max_iter=10000), {"alpha": [0.01, 0.1, 1.0, 10.0]}),
    "KNN": (KNeighborsRegressor(), {"n_neighbors": [3, 5, 7, 9]}),
    "SVR": (SVR(), {"C": [1.0, 10.0, 100.0], "gamma": ["scale", 0.01, 0.1]}),
    "RF": (
        RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
        {"n_estimators": [100, 200], "max_depth": [None, 8, 12]},
    ),
    "GBR": (
        GradientBoostingRegressor(random_state=RANDOM_SEED),
        {"n_estimators": [80, 120], "learning_rate": [0.05, 0.1]},
    ),
    "XGB": (
        xgb.XGBRegressor(
            objective="reg:squarederror", random_state=RANDOM_SEED, n_jobs=-1
        ),
        {"n_estimators": [80, 120], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
    ),
}


def _best_estimator(name: str, X_tr, y_tr, verbose: bool = True):
    """Run grid search with 5-fold CV for a single model."""
    estimator, grid = MODEL_GRIDS[name]
    n_splits = min(CV_FOLDS, int(np.floor(len(X_tr) / 2)))
    if "n_neighbors" in grid:
        # cap K so it stays valid for the smallest CV fold
        max_k = max(1, len(X_tr) - len(X_tr) // n_splits - 1)
        grid = {"n_neighbors": [k for k in grid["n_neighbors"] if k <= max_k] or [max_k]}
    if not grid:
        estimator.fit(X_tr, y_tr)
        return estimator
    search = GridSearchCV(
        estimator,
        grid,
        cv=KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED),
        scoring="r2",
        n_jobs=-1,
    )
    search.fit(X_tr, y_tr)
    if verbose:
        print(f"    {name:4s} best_params={search.best_params_}")
    return search.best_estimator_


def benchmark_ml_models(
    X_num_tr: np.ndarray,
    y_tr: np.ndarray,
    X_num_te: np.ndarray,
    y_te: np.ndarray,
    models: Optional[List[str]] = None,
    scale: bool = True,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """Train and evaluate the conventional ML baseline models.

    Args:
        X_num_tr / X_num_te: raw numerical feature matrices.
        y_tr / y_te: target values.
        models: subset of model names; defaults to all eight.
        scale: z-score the numerical features (needed by distance-based
            models such as KNN and SVR).
        verbose: print tuning progress.

    Returns:
        ``{model_name: {"model": estimator, "train_metrics": ..., "test_metrics": ...}}``
    """
    models = models or list(MODEL_GRIDS.keys())
    scaler = StandardScaler()
    if scale:
        X_tr = scaler.fit_transform(X_num_tr).astype(np.float32)
        X_te = scaler.transform(X_num_te).astype(np.float32)
    else:
        X_tr, X_te = X_num_tr, X_num_te

    results: Dict[str, Dict] = {}
    for name in models:
        if verbose:
            print(f"  [{name}]")
        est = _best_estimator(name, X_tr, y_tr, verbose=verbose)
        train_pred = est.predict(X_tr)
        test_pred = est.predict(X_te)
        results[name] = {
            "model": est,
            "train_metrics": calc_metrics(y_tr, train_pred),
            "test_metrics": calc_metrics(y_te, test_pred),
        }
    return results


def summary_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """Turn the benchmark result dict into a comparison table."""
    rows = []
    for name, res in results.items():
        row = {"model": name}
        row.update({f"test_{k}": round(v, 3) for k, v in res["test_metrics"].items()})
        rows.append(row)
    return pd.DataFrame(rows)
