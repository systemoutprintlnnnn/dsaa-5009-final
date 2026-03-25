"""Data preprocessing utilities for length-controllable multi-task learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

SHORT_MAX = 15
MEDIUM_MAX = 35


@dataclass(frozen=True)
class BucketConfig:
    short_max: int = SHORT_MAX
    medium_max: int = MEDIUM_MAX


def get_length_bucket(summary: str, config: BucketConfig = BucketConfig()) -> str:
    word_count = len(summary.split())
    if word_count <= config.short_max:
        return "SHORT"
    if word_count <= config.medium_max:
        return "MEDIUM"
    return "LONG"


def get_length_token(summary: str, config: BucketConfig = BucketConfig()) -> str:
    return f"<len_{get_length_bucket(summary, config)}>"


def build_multitask_samples(sample: Dict[str, str], config: BucketConfig = BucketConfig()) -> List[Dict[str, str]]:
    length_token = get_length_token(sample["summary"], config)

    summarize_sample = {
        "task": "summarize",
        "input": f"{length_token} [SUMMARIZE] {sample['dialogue']}",
        "target": sample["summary"],
        "length_token": length_token,
        "length_bucket": get_length_bucket(sample["summary"], config),
    }

    topic_sample = {
        "task": "topic",
        "input": f"[TOPIC] {sample['dialogue']}",
        "target": sample["topic"],
        "length_token": None,
        "length_bucket": None,
    }

    return [summarize_sample, topic_sample]
