#!/usr/bin/env python3
"""Analyze DialogSum dataset statistics and validate length buckets."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

SHORT_MAX = 15
MEDIUM_MAX = 35


def get_bucket(word_count: int) -> str:
    if word_count <= SHORT_MAX:
        return "SHORT"
    if word_count <= MEDIUM_MAX:
        return "MEDIUM"
    return "LONG"


def percentile(sorted_values: list[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def summarize_split(dataset_split):
    lengths = [len(sample["summary"].split()) for sample in dataset_split]
    sorted_lengths = sorted(lengths)
    bucket_counts = {"SHORT": 0, "MEDIUM": 0, "LONG": 0}
    for length in lengths:
        bucket_counts[get_bucket(length)] += 1

    total = len(lengths)
    bucket_ratios = {
        key: round(value / total, 4) if total else 0.0
        for key, value in bucket_counts.items()
    }

    return {
        "num_samples": total,
        "length_stats": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": round(mean(lengths), 2),
            "median": round(median(lengths), 2),
            "p25": round(percentile(sorted_lengths, 0.25), 2),
            "p75": round(percentile(sorted_lengths, 0.75), 2),
            "p90": round(percentile(sorted_lengths, 0.90), 2),
            "p95": round(percentile(sorted_lengths, 0.95), 2),
        },
        "bucket_counts": bucket_counts,
        "bucket_ratios": bucket_ratios,
    }


def save_bucket_plot(report: dict) -> Path:
    splits = list(report["splits"].keys())
    bucket_names = ["SHORT", "MEDIUM", "LONG"]
    colors = {"SHORT": "#4CAF50", "MEDIUM": "#2196F3", "LONG": "#FF9800"}

    x = range(len(splits))
    width = 0.22

    plt.figure(figsize=(10, 6))
    for idx, bucket in enumerate(bucket_names):
        values = [report["splits"][split]["bucket_ratios"][bucket] for split in splits]
        positions = [i + (idx - 1) * width for i in x]
        plt.bar(positions, values, width=width, label=bucket, color=colors[bucket])

    plt.xticks(list(x), splits)
    plt.ylim(0, 1.0)
    plt.ylabel("Ratio")
    plt.title("DialogSum Summary Length Bucket Distribution")
    plt.legend()
    plt.tight_layout()

    output_path = METRICS_DIR / "length_distribution.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def main() -> None:
    print("Loading DialogSum dataset...")
    dataset = load_dataset("knkarthick/dialogsum")

    report = {
        "dataset": "knkarthick/dialogsum",
        "bucket_definition": {
            "SHORT": f"1-{SHORT_MAX} words",
            "MEDIUM": f"{SHORT_MAX + 1}-{MEDIUM_MAX} words",
            "LONG": f">={MEDIUM_MAX + 1} words",
        },
        "splits": {},
    }

    for split_name in ["train", "validation", "test"]:
        print(f"Analyzing split: {split_name}")
        split_report = summarize_split(dataset[split_name])
        report["splits"][split_name] = split_report

        print(f"\n[{split_name}]")
        print(f"samples: {split_report['num_samples']}")
        print(f"mean length: {split_report['length_stats']['mean']}")
        print(f"median length: {split_report['length_stats']['median']}")
        print(f"bucket ratios: {split_report['bucket_ratios']}")

    output_path = METRICS_DIR / "data_stats.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved stats to: {output_path}")

    plot_path = save_bucket_plot(report)
    print(f"Saved plot to: {plot_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
