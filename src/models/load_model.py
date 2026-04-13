"""Model loading utilities for dialogue summarization experiments."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer

SPECIAL_TOKENS = [
    "<len_SHORT>",
    "<len_MEDIUM>",
    "<len_LONG>",
    "[SUMMARIZE]",
    "[TOPIC]",
]

# LoRA target modules per architecture
FLAN_T5_TARGET_MODULES = ["q", "v"]
QWEN_TARGET_MODULES = ["q_proj", "v_proj"]


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "google/flan-t5-base"
    special_tokens: List[str] | None = None
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] | None = None

    def resolved_special_tokens(self) -> List[str]:
        return self.special_tokens or SPECIAL_TOKENS

    def resolved_target_modules(self) -> List[str]:
        return self.target_modules or ["q", "v"]


def load_tokenizer(config: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    added = tokenizer.add_tokens(config.resolved_special_tokens())
    return tokenizer, added


def load_base_model(config: ModelConfig):
    return AutoModelForSeq2SeqLM.from_pretrained(config.model_name)


def build_lora_config(config: ModelConfig) -> LoraConfig:
    return LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.resolved_target_modules(),
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )


def prepare_model(config: ModelConfig | None = None):
    config = config or ModelConfig()
    tokenizer, added_tokens = load_tokenizer(config)
    model = load_base_model(config)
    original_vocab = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    resized_vocab = model.get_input_embeddings().weight.shape[0]

    lora_config = build_lora_config(config)
    peft_model = get_peft_model(model, lora_config)

    trainable_params = 0
    total_params = 0
    for _, param in peft_model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    report = {
        "model_name": config.model_name,
        "added_tokens": added_tokens,
        "special_tokens": config.resolved_special_tokens(),
        "original_vocab_size": original_vocab,
        "resized_vocab_size": resized_vocab,
        "lora": {
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "target_modules": config.resolved_target_modules(),
        },
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "trainable_ratio": round(trainable_params / total_params, 6) if total_params else 0.0,
        "config": asdict(config),
    }

    return tokenizer, peft_model, report


def prepare_causal_model(config: ModelConfig | None = None):
    """Prepare a decoder-only causal LM (e.g. Qwen) with LoRA."""
    config = config or ModelConfig(
        model_name="Qwen/Qwen3.5-0.8B",
        target_modules=QWEN_TARGET_MODULES,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    added = tokenizer.add_tokens(config.resolved_special_tokens())

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(len(tokenizer))

    lora_cfg = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.resolved_target_modules(),
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, lora_cfg)

    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())

    report = {
        "model_name": config.model_name,
        "model_type": "causal",
        "added_tokens": added,
        "special_tokens": config.resolved_special_tokens(),
        "lora": {
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "target_modules": config.resolved_target_modules(),
        },
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "trainable_ratio": round(trainable_params / total_params, 6) if total_params else 0.0,
    }

    return tokenizer, peft_model, report
