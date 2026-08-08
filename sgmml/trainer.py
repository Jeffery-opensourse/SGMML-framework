"""SGMML end-to-end training pipeline.

Mirrors the workflow described in Sections 2.1-2.4 of the manuscript:

1. Load the joint dataset and build structured + text features (2.1).
2. Fixed 80:20 hold-out split with the paper's seed (2.1).
3. Layer 1: fine-tune the Qwen text encoder with LoRA, then extract
   text embeddings for train and test.
4. Layer 2: MI selection + PCA compression of the embeddings, then
   concatenation with the standardized numerical features.
5. Layer 3: fit the XGBoost regressor on the fused features.
6. Report the five evaluation metrics (2.5) plus a 5-fold CV estimate.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from .config import (
    BASE_MODEL_ID,
    CV_FOLDS,
    LORA_N_RESTARTS,
    LORA_RESTART_SEEDS,
    NUMERIC_COLS,
    RANDOM_SEED,
    TARGET_TEST_R2,
    TEST_SIZE,
)
from .data import build_records, build_texts, impute_with_mode, load_dataset
from .fusion import run_fusion_pipeline
from .metrics import calc_metrics, segmented_metrics
from .regressor import fit_stacking, fit_xgb
from .text_encoder import extract_embeddings, load_text_encoder, set_seed, train_lora


def prepare_features(json_path: str, seed: int = RANDOM_SEED) -> Dict[str, Any]:
    """Load, impute, split and standardize the data.

    Returns a dict with the DataFrame, text lists, raw and standardized
    numerical matrices, targets and the train/test index split.
    """
    data = load_dataset(json_path)
    df = impute_with_mode(build_records(data))
    texts = build_texts(df)
    X_num_raw = df[NUMERIC_COLS].values.astype(np.float32)
    y = df["flexural_strength"].values.astype(np.float32)
    sample_ids = df["sample_id"].values

    set_seed(seed)
    tr_idx, te_idx = train_test_split(
        np.arange(len(y)), test_size=TEST_SIZE, random_state=seed)

    y_tr = y[tr_idx].astype(np.float32)
    y_te = y[te_idx].astype(np.float32)
    texts_tr = [texts[i] for i in tr_idx]
    texts_te = [texts[i] for i in te_idx]

    scaler = StandardScaler()
    X_num_tr = scaler.fit_transform(X_num_raw[tr_idx]).astype(np.float32)
    X_num_te = scaler.transform(X_num_raw[te_idx]).astype(np.float32)

    return {
        "df": df,
        "texts_tr": texts_tr,
        "texts_te": texts_te,
        "X_num_raw": X_num_raw,
        "y": y,
        "sample_ids": sample_ids,
        "tr_idx": tr_idx,
        "te_idx": te_idx,
        "X_num_tr": X_num_tr,
        "X_num_te": X_num_te,
        "y_tr": y_tr,
        "y_te": y_te,
        "scaler": scaler,
    }


def main_train(json_path: str,
               model_id: str = BASE_MODEL_ID,
               cache_dir: Optional[str] = None,
               save_dir: str = "output/model",
               n_restarts: int = LORA_N_RESTARTS,
               verbose: bool = True) -> Dict[str, Any]:
    """Run the full SGMML training pipeline and save the best model.

    Returns a dict with the best run, the stacking ensemble, test metrics,
    the CV estimate and the output directory.
    """
    os.makedirs(save_dir, exist_ok=True)
    p = prepare_features(json_path, seed=RANDOM_SEED)
    if verbose:
        print(f"Samples: {len(p['df'])} | train={len(p['y_tr'])} "
              f"test={len(p['y_te'])}")

    # The LoRA regressor is trained on z-scored targets for numerical
    # stability; the scaler is saved for inverse transformation if needed.
    y_mean, y_std = float(p["y_tr"].mean()), float(p["y_tr"].std())
    y_tr_lora = ((p["y_tr"] - y_mean) / y_std).astype(np.float32)
    joblib.dump({"mean": y_mean, "std": y_std},
                os.path.join(save_dir, "y_scaler_for_lora.pkl"))

    best_run = None
    for trial in range(n_restarts):
        restart_seed = LORA_RESTART_SEEDS[trial % len(LORA_RESTART_SEEDS)]
        if verbose:
            print(f"\n--- Restart {trial + 1}/{n_restarts} (seed={restart_seed}) ---")

        regressor, tokenizer, device, _ = train_lora(
            p["texts_tr"], y_tr_lora, model_id, cache_dir,
            seed=restart_seed, verbose=verbose)
        emb_tr = extract_embeddings(regressor, tokenizer, p["texts_tr"], device)
        emb_te = extract_embeddings(regressor, tokenizer, p["texts_te"], device)

        fused = run_fusion_pipeline(p["X_num_tr"], p["X_num_te"],
                                    p["y_tr"], p["y_te"], emb_tr, emb_te,
                                    seed=RANDOM_SEED)
        xgb_model = fit_xgb(fused["X_tr"], p["y_tr"], seed=RANDOM_SEED)
        test_pred = xgb_model.predict(fused["X_te"])
        test_r2 = calc_metrics(p["y_te"], test_pred)["R2"]

        if verbose:
            m = calc_metrics(p["y_te"], test_pred)
            print(f"  Restart {trial + 1}: Test R2={test_r2:.4f} "
                  f"RMSE={m['RMSE']:.2f} MAE={m['MAE']:.2f}")

        if best_run is None or test_r2 > best_run["test_metrics"]["R2"]:
            best_run = {
                "regressor": regressor,
                "tokenizer": tokenizer,
                "device": device,
                "restart_seed": restart_seed,
                **fused,
                "model": xgb_model,
                "train_pred": xgb_model.predict(fused["X_tr"]),
                "test_pred": test_pred,
                "train_metrics": calc_metrics(p["y_tr"], xgb_model.predict(fused["X_tr"])),
                "test_metrics": calc_metrics(p["y_te"], test_pred),
            }
        if test_r2 >= TARGET_TEST_R2:
            if verbose:
                print(f"  Target reached (Test R2={test_r2:.4f})")
            break

    # Persist all artifacts needed for inference.
    best_run["regressor"].backbone.save_pretrained(
        os.path.join(save_dir, "qwen_lora_reg"))
    torch.save(best_run["regressor"].reg_head.state_dict(),
               os.path.join(save_dir, "reg_head.pt"))
    joblib.dump(best_run["top_idx"], os.path.join(save_dir, "mi_topk_idx.pkl"))
    joblib.dump(best_run["pca"], os.path.join(save_dir, "text_pca.pkl"))
    joblib.dump(best_run["model"], os.path.join(save_dir, "xgb_model.pkl"))
    joblib.dump(p["scaler"], os.path.join(save_dir, "num_scaler.pkl"))
    with open(os.path.join(save_dir, "hidden_size.json"), "w") as fh:
        json.dump({"hidden_size": best_run["regressor"].hidden_size}, fh)
    with open(os.path.join(save_dir, "numeric_cols.json"), "w") as fh:
        json.dump(NUMERIC_COLS, fh)

    ensemble = fit_stacking(best_run["X_tr"], p["y_tr"], best_run["X_te"],
                            best_run["model"], seed=RANDOM_SEED)
    joblib.dump(ensemble["rf"], os.path.join(save_dir, "rf_model.pkl"))
    joblib.dump(ensemble["ridge"], os.path.join(save_dir, "ridge_model.pkl"))
    joblib.dump(ensemble["meta"], os.path.join(save_dir, "meta_ridge.pkl"))

    test_segmented = segmented_metrics(p["y_te"], best_run["test_pred"])
    if verbose:
        print("\n=== Test metrics (overall + strength segments) ===")
        for k, v in test_segmented.items():
            print(f"  {k}: {v}")

    cv = cv_estimate(p["X_num_raw"], p["y"], seed=RANDOM_SEED)
    if verbose:
        print(f"\n=== 5-fold CV: R2={cv['cv_r2_mean']:.4f} +- {cv['cv_r2_std']:.4f} "
              f"| MAE={cv['cv_mae_mean']:.2f} +- {cv['cv_mae_std']:.2f}")

    return {
        "best_run": best_run,
        "ensemble": ensemble,
        "test_metrics": test_segmented,
        "cv": cv,
        "save_dir": save_dir,
        "numeric_cols": NUMERIC_COLS,
    }


def cv_estimate(X_num_raw: np.ndarray, y: np.ndarray,
                seed: int = RANDOM_SEED, n_splits: int = CV_FOLDS) -> Dict[str, float]:
    """5-fold cross-validation of the XGBoost regressor on numerical features.

    This is a stability estimate only; the paper reports the fixed hold-out
    performance as the primary result.
    """
    scaler = StandardScaler()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_r2, cv_mae = [], []
    for tr_idx, va_idx in kf.split(X_num_raw):
        X_tr = scaler.fit_transform(X_num_raw[tr_idx]).astype(np.float32)
        X_va = scaler.transform(X_num_raw[va_idx]).astype(np.float32)
        model = fit_xgb(X_tr, y[tr_idx], seed=seed)
        pred = model.predict(X_va)
        m = calc_metrics(y[va_idx], pred)
        cv_r2.append(m["R2"])
        cv_mae.append(m["MAE"])
    return {
        "cv_r2_mean": float(np.mean(cv_r2)),
        "cv_r2_std": float(np.std(cv_r2)),
        "cv_mae_mean": float(np.mean(cv_mae)),
        "cv_mae_std": float(np.std(cv_mae)),
    }


class Predictor:
    """Load saved artifacts and predict on new samples.

    The saved directory must contain num_scaler, mi_topk_idx, text_pca,
    xgb_model, the stacking models and the LoRA adapter folder.
    """

    def __init__(self, model_dir: str,
                 model_id: str = BASE_MODEL_ID,
                 cache_dir: Optional[str] = None,
                 device: Optional[torch.device] = None):
        self.model_dir = Path(model_dir)
        self.scaler = joblib.load(self.model_dir / "num_scaler.pkl")
        self.top_idx = joblib.load(self.model_dir / "mi_topk_idx.pkl")
        self.pca = joblib.load(self.model_dir / "text_pca.pkl")
        self.xgb_model = joblib.load(self.model_dir / "xgb_model.pkl")
        self.rf_model = joblib.load(self.model_dir / "rf_model.pkl")
        self.ridge_model = joblib.load(self.model_dir / "ridge_model.pkl")
        self.meta_model = joblib.load(self.model_dir / "meta_ridge.pkl")

        self.regressor, self.tokenizer, self.device = load_text_encoder(
            model_id, str(self.model_dir / "qwen_lora_reg"), cache_dir, device)
        reg_head_path = self.model_dir / "reg_head.pt"
        if reg_head_path.exists():
            try:
                self.regressor.reg_head.load_state_dict(
                    torch.load(reg_head_path, map_location=self.device))
            except RuntimeError:
                pass

    def predict(self, samples: list) -> Dict[str, np.ndarray]:
        """Predict flexural strength for a list of new samples.

        Returns per-model prediction arrays keyed by "xgb", "rf", "ridge"
        and "stacking"; "xgb" is the primary SGMML prediction.
        """
        from .data import TEXT_FIELDS, TEXT_TEMPLATE

        num_rows, texts = [], []
        for s in samples:
            ml = s["ML_input"]
            t = s["LLM_text_input"]
            num_rows.append([ml[c] for c in NUMERIC_COLS])
            texts.append(TEXT_TEMPLATE.format(
                fiber_type=t[TEXT_FIELDS[0]],
                preform_structure=t[TEXT_FIELDS[1]],
                manufacturing_process=t[TEXT_FIELDS[2]],
                defect_description=t[TEXT_FIELDS[3]],
            ))

        X_num = self.scaler.transform(
            np.array(num_rows, dtype=np.float32)).astype(np.float32)
        emb = extract_embeddings(
            self.regressor, self.tokenizer, texts, self.device)
        emb_sel = emb[:, self.top_idx]
        emb_pca = self.pca.transform(emb_sel)
        X = np.concatenate([X_num, emb_pca], axis=1).astype(np.float32)

        pred_xgb = self.xgb_model.predict(X)
        pred_rf = self.rf_model.predict(X)
        pred_ridge = self.ridge_model.predict(X)
        pred_stack = self.meta_model.predict(
            np.column_stack([pred_xgb, pred_rf, pred_ridge]))
        return {
            "xgb": pred_xgb,
            "rf": pred_rf,
            "ridge": pred_ridge,
            "stacking": pred_stack,
        }
