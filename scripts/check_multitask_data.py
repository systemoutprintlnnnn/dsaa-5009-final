#!/usr/bin/env python3
"""Validate multitask data construction for DialogSum."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

from src.data.preprocessing import build_multitask_samples

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("Loading DialogSum train split...")
    dataset = load_dataset("knkarthick/dialogsum", split="train")

    sample_indexes = [0, 1, 2, 100, 999]
    checked_samples = []

    for idx in sample_indexes:
        original = dataset[idx]
        multitask = build_multitask_samples(original)
        checked_samples.append(
            {
                "index": idx,
                "dialogue_preview": original["dialogue"][:160],
                "summary": original["summary"],
                "topic": original["topic"],
                "multitask_samples": multitask,
            }
        )

    # quick aggregate sanity check
    first_200 = dataset.select(range(200))
    summary_count = 0
    topic_count = 0
    bucket_counts = {"SHORT": 0, "MEDIUM": 0, "LONG": 0}

    for row in first_200:
        samples = build_multitask_samples(row)
        for sample in samples:
            if sample["task"] == "summarize":
                summary_count += 1
                bucket_counts[sample["length_bucket"]] += 1
            elif sample["task"] == "topic":
                topic_count += 1

    report = {
        "checked_indexes": sample_indexes,
        "num_examples_checked": len(sample_indexes),
        "aggregate_check_on_first_200": {
            "summarize_samples": summary_count,
            "topic_samples": topic_count,
            "bucket_counts": bucket_counts,
        },
        "examples": checked_samples,
    }

    output_path = METRICS_DIR / "multitask_samples.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("Multitask sample construction looks valid.")
    print(f"Saved report to: {output_path}")
    print(f"Aggregate sample counts: summarize={summary_count}, topic={topic_count}")
    print(f"Bucket counts (first 200 summarize tasks): {bucket_counts}")


if __name__ == "__main__":
    main()
