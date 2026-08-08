#!/usr/bin/env python
"""Command-line entry point for SGMML training.

Usage:
    python scripts/train.py --data data/example_samples.json \
        --save_dir output/model --n_restarts 5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sgmml.trainer import main_train  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train the SGMML framework")
    parser.add_argument("--data", required=True,
                        help="Path to the dataset JSON")
    parser.add_argument("--model_id", default="qwen/Qwen3.5-0.8B",
                        help="HuggingFace/ModelScope base model id")
    parser.add_argument("--cache_dir", default=None,
                        help="Local cache directory for the base model")
    parser.add_argument("--save_dir", default="output/model",
                        help="Directory where artifacts are saved")
    parser.add_argument("--n_restarts", type=int, default=5,
                        help="Number of LoRA restarts to try")
    return parser.parse_args()


def main():
    args = parse_args()
    result = main_train(args.data, args.model_id, args.cache_dir,
                        args.save_dir, args.n_restarts)
    print(f"\nDone. Model artifacts saved to {args.save_dir}")
    print(f"Test metrics: {result['test_metrics']['overall']}")


if __name__ == "__main__":
    main()
