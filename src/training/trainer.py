"""Minimal training utilities for CP-05 smoke test."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from transformers import PreTrainedTokenizerBase

MAX_INPUT_LENGTH = 384
MAX_TARGET_LENGTH = 96
DEFAULT_LR = 5e-4


@dataclass(frozen=True)
class TrainingSmokeConfig:
    max_input_length: int = MAX_INPUT_LENGTH
    max_target_length: int = MAX_TARGET_LENGTH
    batch_size: int = 1
    learning_rate: float = DEFAULT_LR


def get_best_available_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


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

    labels = tokenizer(
        text_target=targets,
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


def run_single_training_step(model, batch: Dict[str, torch.Tensor], lr: float = DEFAULT_LR) -> Dict[str, float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer.zero_grad(set_to_none=True)

    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    grad_norm_sq = 0.0
    trainable_grad_params = 0
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm_sq += float(torch.norm(param.grad.detach(), p=2).item() ** 2)
            trainable_grad_params += 1

    return {
        "loss": float(loss.detach().cpu().item()),
        "grad_norm": grad_norm_sq ** 0.5,
        "trainable_grad_params": trainable_grad_params,
    }
