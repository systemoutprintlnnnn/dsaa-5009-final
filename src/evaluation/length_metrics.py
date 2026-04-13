"""Length control metrics: Length Accuracy and Length MAE."""

from __future__ import annotations

from typing import Dict, List, Tuple

SHORT_RANGE = (5, 15)
MEDIUM_RANGE = (16, 35)
LONG_RANGE = (36, 999)

BUCKET_RANGES: Dict[str, Tuple[int, int]] = {
    "SHORT": SHORT_RANGE,
    "MEDIUM": MEDIUM_RANGE,
    "LONG": LONG_RANGE,
}

LEN_TOKEN_TO_BUCKET = {
    "<len_SHORT>": "SHORT",
    "<len_MEDIUM>": "MEDIUM",
    "<len_LONG>": "LONG",
}


def length_accuracy(
    predictions: List[str],
    target_buckets: List[str],
) -> float:
    """Fraction of predictions whose word count falls in the target bucket range."""
    correct = 0
    for pred, bucket in zip(predictions, target_buckets):
        word_count = len(pred.split())
        lo, hi = BUCKET_RANGES[bucket]
        if lo <= word_count <= hi:
            correct += 1
    return round(correct / len(predictions), 4) if predictions else 0.0


def length_mae(
    predictions: List[str],
    target_buckets: List[str],
) -> float:
    """Mean absolute error between predicted word count and bucket midpoint."""
    errors = []
    for pred, bucket in zip(predictions, target_buckets):
        word_count = len(pred.split())
        lo, hi = BUCKET_RANGES[bucket]
        midpoint = (lo + hi) / 2
        errors.append(abs(word_count - midpoint))
    return round(sum(errors) / len(errors), 2) if errors else 0.0


def classify_length(text: str) -> str:
    """Classify text into SHORT / MEDIUM / LONG bucket by word count."""
    wc = len(text.split())
    if wc <= SHORT_RANGE[1]:
        return "SHORT"
    if wc <= MEDIUM_RANGE[1]:
        return "MEDIUM"
    return "LONG"
