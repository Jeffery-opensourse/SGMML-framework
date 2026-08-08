#!/usr/bin/env python
"""Command-line entry point for SGMML prediction on new samples.

Usage:
    python scripts/predict.py --model_dir output/model \
        --data data/new_samples.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sgmml.data import load_dataset  # noqa: E402
from sgmml.trainer import Predictor  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict flexural strength with a trained SGMML model")
    parser.add_argument("--model_dir", required=True,
                        help="Directory with trained artifacts")
    parser.add_argument("--data", required=True,
                        help="Path to the new-samples JSON")
    parser.add_argument("--model_id", default="qwen/Qwen3.5-0.8B",
                        help="Base model id matching the LoRA adapter")
    parser.add_argument("--cache_dir", default=None,
                        help="Local cache directory for the base model")
    return parser.parse_args()


def main():
    args = parse_args()
    samples = load_dataset(args.data)
    predictor = Predictor(args.model_dir, args.model_id, args.cache_dir)
    preds = predictor.predict(samples)

    names = [s.get("sample_name", s.get("sample_id", f"S{i}"))
             for i, s in enumerate(samples)]
    header = f"{'sample':<18}{'xgb':>10}{'rf':>10}{'ridge':>10}{'stacking':>12}"
    print(header)
    print("-" * len(header))
    for i, name in enumerate(names):
        print(f"{name:<18}"
              f"{preds['xgb'][i]:>10.2f}"
              f"{preds['rf'][i]:>10.2f}"
              f"{preds['ridge'][i]:>10.2f}"
              f"{preds['stacking'][i]:>12.2f}")


if __name__ == "__main__":
    main()
