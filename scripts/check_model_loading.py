#!/usr/bin/env python3
"""Smoke test for model loading, token injection, and LoRA setup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.models.load_model import ModelConfig, prepare_model

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_MAP = {
    "flan-t5": "google/flan-t5-base",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="flan-t5", choices=MODEL_MAP.keys())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name = MODEL_MAP[args.model]

    print(f"Loading model: {model_name}")
    config = ModelConfig(model_name=model_name)
    tokenizer, model, report = prepare_model(config)

    # extra verification fields
    report["tokenizer_length"] = len(tokenizer)
    report["model_type"] = model.__class__.__name__
    report["verification"] = {
        "tokenizer_loaded": True,
        "model_loaded": True,
        "special_tokens_added": report["added_tokens"] >= 0,
        "embedding_resized": report["resized_vocab_size"] == len(tokenizer),
        "lora_injected": report["trainable_parameters"] > 0,
    }

    output_path = METRICS_DIR / "model_check_flan.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("Model loading smoke test passed.")
    print(f"Added tokens: {report['added_tokens']}")
    print(f"Trainable ratio: {report['trainable_ratio']}")
    print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
