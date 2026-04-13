"""ROUGE evaluation for dialogue summarization."""

from __future__ import annotations

from typing import Dict, List

import evaluate


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum."""
    rouge = evaluate.load("rouge")
    results = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )
    return {k: round(v * 100, 2) for k, v in results.items()}
