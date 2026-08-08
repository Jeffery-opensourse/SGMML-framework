"""Layer 3 - XGBoost regression prediction.

The fused feature vector is fed into an XGBoost regressor that builds an
additive ensemble of regression trees. An optional XGB+RF+Ridge stacking
ensemble is also provided; the single XGBoost model is the primary
predictor reported in the paper.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold, cross_val_predict

from .config import RANDOM_SEED, XGB_PARAMS


def fit_xgb(X_tr: np.ndarray, y_tr: np.ndarray,
            seed: int = RANDOM_SEED) -> xgb.XGBRegressor:
    """Train the XGBoost regressor on the fused features."""
    model = xgb.XGBRegressor(
        objective="reg:squarederror", random_state=seed, n_jobs=-1, **XGB_PARAMS
    )
    model.fit(X_tr, y_tr)
    return model


def fit_stacking(X_tr, y_tr, X_te, xgb_model, seed: int = RANDOM_SEED) -> Dict:
    """Build the XGB + RF + Ridge stacking ensemble.

    Out-of-fold predictions on the training set train the Ridge meta-learner;
    test predictions combine the three base models through the meta-learner.
    """
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=3,
        random_state=seed, n_jobs=-1,
    )
    ridge = Ridge(alpha=5.0, random_state=seed)

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof_xgb = cross_val_predict(xgb_model, X_tr, y_tr, cv=kf, n_jobs=-1)
    oof_rf = cross_val_predict(rf, X_tr, y_tr, cv=kf, n_jobs=-1)
    oof_ridge = cross_val_predict(ridge, X_tr, y_tr, cv=kf, n_jobs=-1)

    rf.fit(X_tr, y_tr)
    ridge.fit(X_tr, y_tr)
    meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    meta.fit(np.column_stack([oof_xgb, oof_rf, oof_ridge]), y_tr)

    pred_xgb = xgb_model.predict(X_te)
    pred_rf = rf.predict(X_te)
    pred_ridge = ridge.predict(X_te)
    pred_meta = meta.predict(np.column_stack([pred_xgb, pred_rf, pred_ridge]))

    return {
        "rf": rf,
        "ridge": ridge,
        "meta": meta,
        "test_pred_xgb": pred_xgb,
        "test_pred_rf": pred_rf,
        "test_pred_ridge": pred_ridge,
        "test_pred_stacking": pred_meta,
    }
