#!/usr/bin/env python
"""Command-line entry point for closed-source LLM predictions.

Usage:
    export OPENAI_API_KEY=...
    export DASHSCOPE_API_KEY=...
    python scripts/predict_api.py --data data/new_samples.json \
        --mode few_shot --train_data data/example_samples.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sgmml.data import load_dataset  # noqa: E402
from sgmml.llm_api import predict_few_shot, predict_zero_shot  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict flexural strength with closed-source LLM APIs")
    parser.add_argument("--data", required=True,
                        help="Path to the new-samples JSON")
    parser.add_argument("--train_data", default=None,
                        help="Path to training data for few-shot demos")
    parser.add_argument("--mode", choices=["zero_shot", "few_shot"],
                        default="few_shot")
    return parser.parse_args()


def main():
    args = parse_args()
    samples = load_dataset(args.data)
    train_samples = load_dataset(args.train_data) if args.train_data else []

    if args.mode == "few_shot":
        results = predict_few_shot(samples, train_samples)
    else:
        results = predict_zero_shot(samples)

    for model, preds in results.items():
        print(f"{model}: {preds}")


if __name__ == "__main__":
    main()
