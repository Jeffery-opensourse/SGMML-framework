"""Example usage of the SGMML package.

This script walks through the package APIs without requiring trained
weights or a GPU. It loads a small sample dataset, inspects the data,
benchmarks the conventional ML baselines, and runs the downstream fusion
pipeline using random placeholder embeddings.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.model_selection import train_test_split

from sgmml.config import RANDOM_SEED, TEST_SIZE
from sgmml.data import (
    build_numeric_matrix,
    build_records,
    build_texts,
    encoding_table,
    impute_with_mode,
    load_dataset,
    missing_value_report,
)
from sgmml.fusion import compression_report, run_fusion_pipeline
from sgmml.metrics import calc_metrics, segmented_metrics
from sgmml.ml_models import benchmark_ml_models, summary_table
from sgmml.regressor import fit_xgb

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "example_samples.json"


def set_seed(seed: int) -> None:
    """Seed numpy only; a full run should also seed torch."""
    np.random.seed(seed)


def main():
    # 1. Load and inspect the data.
    data = load_dataset(DATA_PATH)
    df = impute_with_mode(build_records(data))
    texts = build_texts(df)
    X_num = build_numeric_matrix(df)
    y = df["flexural_strength"].values.astype(np.float32)

    print(f"Samples: {len(df)}")
    print("\nMissing value report:")
    print(missing_value_report(df).to_string(index=False))

    print("\nManufacturing process encoding:")
    print(encoding_table(df)["manufacturing_process_code"])

    # 2. The ML steps below need a few samples with complete numerical
    #    descriptors. The bundled example only demonstrates the data format,
    #    so we skip them gracefully when that is the case.
    if len(df) < 5 or np.isnan(X_num).any():
        print("\n(Not enough samples for the ML pipeline; "
              "the bundled data only shows the input format.)")
        return

    # 3. Fixed 80:20 split, as in the paper.
    set_seed(RANDOM_SEED)
    tr_idx, te_idx = train_test_split(
        np.arange(len(y)), test_size=TEST_SIZE, random_state=RANDOM_SEED)
    X_tr, X_te = X_num[tr_idx], X_num[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    texts_tr, texts_te = [texts[i] for i in tr_idx], [texts[i] for i in te_idx]

    # 4. Benchmark the conventional ML baselines.
    print("\nConventional ML benchmark (test set):")
    ml_results = benchmark_ml_models(X_tr, y_tr, X_te, y_te, verbose=False)
    print(summary_table(ml_results).to_string(index=False))

    # 5. Downstream fusion with placeholder embeddings. In a real run the
    #    embeddings come from the LoRA-fine-tuned Qwen encoder (Layer 1).
    rng = np.random.default_rng(42)
    emb_tr = rng.standard_normal((len(tr_idx), 1024), dtype=np.float32)
    emb_te = rng.standard_normal((len(te_idx), 1024), dtype=np.float32)

    fused = run_fusion_pipeline(X_tr, X_te, y_tr, y_te, emb_tr, emb_te,
                                seed=RANDOM_SEED)
    print("\nFeature compression report:")
    print(compression_report(emb_tr, y_tr, seed=RANDOM_SEED))

    model = fit_xgb(fused["X_tr"], y_tr, seed=RANDOM_SEED)
    pred_te = model.predict(fused["X_te"])
    print("\nFused model test metrics:")
    print(calc_metrics(y_te, pred_te))
    print("\nSegmented test metrics:")
    print(segmented_metrics(y_te, pred_te))


if __name__ == "__main__":
    main()
