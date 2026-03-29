#!/usr/bin/env python3
"""CP-05 smoke test: tokenize a tiny batch and run 1 forward/backward step."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import load_dataset

from src.data.preprocessing import build_multitask_samples
from src.models.load_model import ModelConfig, prepare_model
from src.training.trainer import (
    TrainingSmokeConfig,
    move_batch_to_device,
    run_single_training_step,
    tokenize_seq2seq_batch,
)

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    smoke_config = TrainingSmokeConfig(batch_size=2)

    dataset = load_dataset("knkarthick/dialogsum", split="train[:2]")
    raw_samples = [dataset[i] for i in range(len(dataset))]

    summarize_samples = []
    for sample in raw_samples:
        multitask_samples = build_multitask_samples(sample)
        summarize_samples.append(multitask_samples[0])

    tokenizer, model, model_report = prepare_model(ModelConfig(model_name="google/flan-t5-base"))

    batch = tokenize_seq2seq_batch(summarize_samples, tokenizer, smoke_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    batch = move_batch_to_device(batch, device)

    step_report = run_single_training_step(model, batch)

    report = {
        "model": "google/flan-t5-base",
        "device": str(device),
        "batch_size": smoke_config.batch_size,
        "input_shape": list(batch["input_ids"].shape),
        "label_shape": list(batch["labels"].shape),
        "loss": step_report["loss"],
        "grad_norm": step_report["grad_norm"],
        "loss_is_finite": bool(torch.isfinite(torch.tensor(step_report["loss"])).item()),
        "used_samples": [
            {
                "input_preview": sample["input"][:120],
                "target_preview": sample["target"][:120],
                "length_bucket": sample["length_bucket"],
            }
            for sample in summarize_samples
        ],
        "model_report": {
            "added_tokens": model_report["added_tokens"],
            "trainable_ratio": model_report["trainable_ratio"],
        },
        "verification": {
            "forward_backward_ok": True,
            "loss_non_nan": bool(torch.isfinite(torch.tensor(step_report["loss"])).item()),
            "batch_tokenized": True,
        },
    }

    output_path = METRICS_DIR / "training_smoke_test.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("Training smoke test passed.")
    print(f"Device: {device}")
    print(f"Loss: {report['loss']:.6f}")
    print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
