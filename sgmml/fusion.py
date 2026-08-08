"""Layer 2 - Feature compression and fusion.

The high-dimensional text embedding is reduced in two steps: mutual
information (MI) selection keeps the top-k dimensions most relevant to
the target, then PCA projects them down under a 90% cumulative explained
variance threshold. The reduced text features are early-concatenated with
the standardized numerical descriptors to form the fused feature vector.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression

from .config import MI_TOPK, PCA_MAX_DIM, PCA_VARIANCE_THRESHOLD


def fit_mi_pca(emb_tr: np.ndarray, y_tr: np.ndarray,
               seed: int = 42, mi_topk: int = MI_TOPK,
               pca_max_dim: int = PCA_MAX_DIM,
               variance_threshold: float = PCA_VARIANCE_THRESHOLD
               ) -> Tuple[np.ndarray, PCA]:
    """Fit MI selection and PCA on the training set only.

    The PCA component count is chosen automatically from the 90%
    cumulative variance threshold and capped at ``pca_max_dim``.
    """
    topk = min(mi_topk, emb_tr.shape[1])
    mi = mutual_info_regression(emb_tr, y_tr, random_state=seed)
    top_idx = np.argsort(mi)[-topk:]
    emb_sel = emb_tr[:, top_idx]

    max_dim = min(pca_max_dim, emb_sel.shape[0], emb_sel.shape[1])
    pca = PCA(n_components=max_dim, random_state=seed)
    pca.fit(emb_sel)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_keep = int(np.searchsorted(cumulative, variance_threshold)) + 1
    n_keep = max(1, min(n_keep, max_dim))

    pca = PCA(n_components=n_keep, random_state=seed)
    pca.fit(emb_sel)
    return top_idx, pca


def fuse_features(X_num: np.ndarray, emb: np.ndarray,
                  top_idx: np.ndarray, pca: PCA) -> np.ndarray:
    """Concatenate the numerical features with the reduced text features."""
    emb_sel = emb[:, top_idx]
    emb_pca = pca.transform(emb_sel)
    return np.concatenate([X_num, emb_pca], axis=1).astype(np.float32)


def run_fusion_pipeline(X_num_tr, X_num_te, y_tr, y_te,
                        emb_tr, emb_te, seed: int = 42) -> Dict:
    """MI -> PCA -> concatenation, returning train/test fused matrices."""
    top_idx, pca = fit_mi_pca(emb_tr, y_tr, seed=seed)
    X_tr = fuse_features(X_num_tr, emb_tr, top_idx, pca)
    X_te = fuse_features(X_num_te, emb_te, top_idx, pca)
    return {"top_idx": top_idx, "pca": pca, "X_tr": X_tr, "X_te": X_te}


def compression_report(emb_tr: np.ndarray, y_tr: np.ndarray,
                       seed: int = 42) -> Dict[str, Optional[float]]:
    """Report how many text dimensions survive MI + PCA compression."""
    top_idx, pca = fit_mi_pca(emb_tr, y_tr, seed=seed)
    return {
        "raw_dim": emb_tr.shape[1],
        "mi_topk": int(len(top_idx)),
        "pca_dim": int(pca.n_components_),
        "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
    }
