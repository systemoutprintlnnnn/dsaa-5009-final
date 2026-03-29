"""Minimal training utilities for CP-05 smoke test."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from transformers import PreTrainedTokenizerBase

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128


@dataclass(frozen=True)
class TrainingSmokeConfig:
    max_input_length: int = MAX_INPUT_LENGTH
    max_target_length: int = MAX_TARGET_LENGTH
    batch_size: int = 2


def tokenize_seq2seq_batch(
    samples: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    config: TrainingSmokeConfig = TrainingSmokeConfig(),
) -> Dict[str, torch.Tensor]:
    inputs = [sample["input"] for sample in samples]
    targets = [sample["target"] for sample in samples]

    model_inputs = tokenizer(
        inputs,
        max_length=config.max_input_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=config.max_target_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

    label_ids = labels["input_ids"]
    label_ids[label_ids == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = label_ids
    return model_inputs


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def run_single_training_step(model, batch: Dict[str, torch.Tensor], lr: float = 1e-3) -> Dict[str, float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer.zero_grad()

    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    grad_norm_sq = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm_sq += float(torch.norm(param.grad.detach(), p=2).item() ** 2)

    return {
        "loss": float(loss.detach().cpu().item()),
        "grad_norm": grad_norm_sq ** 0.5,
    }
