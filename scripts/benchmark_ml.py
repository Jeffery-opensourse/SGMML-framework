#!/usr/bin/env python
"""Benchmark the eight conventional ML models on structured descriptors.

Usage:
    python scripts/benchmark_ml.py --data data/example_samples.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sgmml.data import (  # noqa: E402
    build_numeric_matrix,
    build_records,
    impute_with_mode,
    load_dataset,
)
from sgmml.ml_models import benchmark_ml_models, summary_table  # noqa: E402
from sgmml.trainer import prepare_features  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark conventional ML models on numerical features")
    parser.add_argument("--data", required=True,
                        help="Path to the dataset JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_dataset(args.data)
    df = impute_with_mode(build_records(data))
    X_num = build_numeric_matrix(df)
    y = df["flexural_strength"].values.astype("float32")

    prepared = prepare_features(args.data)
    results = benchmark_ml_models(
        prepared["X_num_tr"], prepared["y_tr"],
        prepared["X_num_te"], prepared["y_te"],
    )
    print(summary_table(results).to_string(index=False))


if __name__ == "__main__":
    main()
