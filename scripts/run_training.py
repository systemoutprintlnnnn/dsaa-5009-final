#!/usr/bin/env python3
"""Full training script for dialogue summarization experiments.

Supports three experiment modes:
  exp0  – Baseline (no length tokens, summarization only)
  exp1  – Length-controllable summarization (single-task)
  exp1_multi – Multi-task: summarization + topic generation

Usage:
  PYTHONPATH=. python scripts/run_training.py --exp exp0
  PYTHONPATH=. python scripts/run_training.py --exp exp1
  PYTHONPATH=. python scripts/run_training.py --exp exp1_multi
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from dataclasses import dataclass as _dataclass
from typing import Any, Dict, List

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainingArguments,
    Trainer,
)

from src.data.preprocessing import BucketConfig, get_length_bucket
from src.models.load_model import SPECIAL_TOKENS, QWEN_TARGET_MODULES, build_lora_config
from peft import get_peft_model, LoraConfig, TaskType

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

# ── Experiment configs ──────────────────────────────────────────────

EXPERIMENTS = {
    # ── FLAN-T5 (Encoder-Decoder / seq2seq) ──
    "exp0": {
        "use_length_tokens": False,
        "multitask": False,
        "output_dir": "results/models/exp0",
        "model_type": "seq2seq",
        "default_model": "google/flan-t5-base",
    },
    "exp1": {
        "use_length_tokens": True,
        "multitask": False,
        "output_dir": "results/models/exp1",
        "model_type": "seq2seq",
        "default_model": "google/flan-t5-base",
    },
    "exp1_multi": {
        "use_length_tokens": True,
        "multitask": True,
        "output_dir": "results/models/exp1_multi",
        "model_type": "seq2seq",
        "default_model": "google/flan-t5-base",
    },
    # ── Qwen (Decoder-Only / causal LM) ──
    "exp0_qwen": {
        "use_length_tokens": False,
        "multitask": False,
        "output_dir": "results/models/exp0_qwen",
        "model_type": "causal",
        "default_model": "Qwen/Qwen3.5-0.8B",
    },
    "exp1_qwen": {
        "use_length_tokens": True,
        "multitask": False,
        "output_dir": "results/models/exp1_qwen",
        "model_type": "causal",
        "default_model": "Qwen/Qwen3.5-0.8B",
    },
    "exp1_multi_qwen": {
        "use_length_tokens": True,
        "multitask": True,
        "output_dir": "results/models/exp1_multi_qwen",
        "model_type": "causal",
        "default_model": "Qwen/Qwen3.5-0.8B",
    },
}

# Natural language length instructions for FLAN-T5
LENGTH_INSTRUCTIONS = {
    "SHORT": "Write a very brief one-sentence summary of the dialogue in 5 to 15 words.",
    "MEDIUM": "Write a short summary of the dialogue in 16 to 35 words.",
    "LONG": "Write a detailed summary of the dialogue in more than 35 words.",
}


# ── Data helpers ────────────────────────────────────────────────────

def preprocess_dataset(dataset_split, tokenizer, exp_cfg, max_input, max_target):
    """Convert raw DialogSum split into tokenized train-ready examples."""

    use_len = exp_cfg["use_length_tokens"]
    multitask = exp_cfg["multitask"]
    bucket_cfg = BucketConfig()

    raw_samples = []

    for row in dataset_split:
        # ── Summarize task ──
        if use_len:
            bucket = get_length_bucket(row["summary"], bucket_cfg)
            length_instr = LENGTH_INSTRUCTIONS[bucket]
            summarize_input = (
                f"Summarize the following dialogue.\n"
                f"Instruction: {length_instr}\n"
                f"Dialogue:\n{row['dialogue']}"
            )
        else:
            summarize_input = (
                f"Summarize the following dialogue.\n"
                f"Dialogue:\n{row['dialogue']}"
            )

        raw_samples.append({
            "input": summarize_input,
            "target": row["summary"],
            "task": "summarize",
        })

        # ── Topic task (multi-task only) ──
        if multitask:
            topic_input = (
                f"What is the topic of the following dialogue? "
                f"Answer in a short phrase.\n"
                f"Dialogue:\n{row['dialogue']}"
            )
            raw_samples.append({
                "input": topic_input,
                "target": row["topic"],
                "task": "topic",
            })

    def tokenize_fn(examples):
        model_inputs = tokenizer(
            examples["input"],
            max_length=max_input,
            truncation=True,
        )
        labels = tokenizer(
            text_target=examples["target"],
            max_length=max_target if examples["task"][0] == "summarize" else 16,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    from datasets import Dataset
    ds = Dataset.from_dict({
        "input": [s["input"] for s in raw_samples],
        "target": [s["target"] for s in raw_samples],
        "task": [s["task"] for s in raw_samples],
    })

    tokenized = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=ds.column_names,
        desc="Tokenizing",
    )
    return tokenized


# ── Causal LM data helpers ──────────────────────────────────────────

def preprocess_dataset_causal(dataset_split, tokenizer, exp_cfg, max_input, max_target):
    """Causal LM: full sequence = prompt + target; labels mask the prompt with -100."""
    use_len = exp_cfg["use_length_tokens"]
    multitask = exp_cfg["multitask"]
    bucket_cfg = BucketConfig()

    all_input_ids, all_attention_mask, all_labels = [], [], []

    for row in dataset_split:
        pairs = []

        # Summarize task
        if use_len:
            bucket = get_length_bucket(row["summary"], bucket_cfg)
            length_instr = LENGTH_INSTRUCTIONS[bucket]
            prompt = (
                f"Summarize the following dialogue.\n"
                f"Instruction: {length_instr}\n"
                f"Dialogue:\n{row['dialogue']}\n"
                f"Summary: "
            )
        else:
            prompt = (
                f"Summarize the following dialogue.\n"
                f"Dialogue:\n{row['dialogue']}\n"
                f"Summary: "
            )
        pairs.append((prompt, row["summary"]))

        if multitask:
            topic_prompt = (
                f"What is the topic of the following dialogue? Answer in a short phrase.\n"
                f"Dialogue:\n{row['dialogue']}\n"
                f"Topic: "
            )
            pairs.append((topic_prompt, row["topic"]))

        for prompt_text, target_text in pairs:
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            target_ids = tokenizer.encode(
                target_text + tokenizer.eos_token, add_special_tokens=False
            )

            # Truncate prompt if combined length exceeds budget
            max_total = max_input + max_target
            if len(prompt_ids) + len(target_ids) > max_total:
                prompt_ids = prompt_ids[-(max_total - len(target_ids)):]

            full_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids
            attn_mask = [1] * len(full_ids)

            all_input_ids.append(full_ids)
            all_attention_mask.append(attn_mask)
            all_labels.append(labels)

    from datasets import Dataset
    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    })


@_dataclass
class CausalLMDataCollator:
    """Right-pad causal LM batches; labels use -100 for padding positions."""
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch: Dict[str, List] = {"input_ids": [], "attention_mask": [], "labels": []}

        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad_len)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad_len)
            batch["labels"].append(f["labels"] + [-100] * pad_len)

        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train dialogue summarization model")
    parser.add_argument("--exp", choices=list(EXPERIMENTS.keys()), required=True, help="Experiment name")
    parser.add_argument("--model_name", default=None, help="HuggingFace model name (default: from experiment config)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 (GPU only)")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps (use with small batch_size to save memory)")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory (causal LM only)")
    parser.add_argument("--output_dir", default=None, help="Override output dir")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit train samples for quick validation")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Limit eval samples to speed up epoch evaluation")
    parser.add_argument("--skip_eval", action="store_true", help="Disable eval during training (avoids MPS memory fragmentation from checkpoint saves)")
    parser.add_argument(
        "--resume_from_checkpoint", default=None,
        help="Resume from a specific checkpoint dir, or 'auto' to continue from the latest checkpoint in output_dir",
    )
    args = parser.parse_args()

    exp_cfg = EXPERIMENTS[args.exp]
    model_type = exp_cfg["model_type"]
    model_name = args.model_name or exp_cfg["default_model"]
    logger.info(
        f"Experiment: {args.exp} | model_type={model_type} | model={model_name} | "
        f"length_tokens={exp_cfg['use_length_tokens']} | multitask={exp_cfg['multitask']}"
    )

    # Determine device
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA GPU")
        args.fp16 = True
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
        args.fp16 = False

    # Output dir
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / exp_cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset (shared for both model types)
    logger.info("Loading DialogSum dataset...")
    ds = load_dataset("knkarthick/dialogsum")
    train_split = ds["train"]
    val_split = ds["validation"]
    if args.max_train_samples:
        train_split = train_split.select(range(min(args.max_train_samples, len(train_split))))
        logger.info(f"Quick validation mode: using {len(train_split)} train samples")
    if args.max_eval_samples:
        val_split = val_split.select(range(min(args.max_eval_samples, len(val_split))))
        logger.info(f"Limiting eval to {len(val_split)} samples")

    # ── FLAN-T5 (seq2seq) branch ────────────────────────────────────
    if model_type == "seq2seq":
        logger.info(f"Loading seq2seq model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if exp_cfg["use_length_tokens"]:
            num_added = tokenizer.add_tokens(SPECIAL_TOKENS)
            logger.info(f"Added {num_added} special tokens")

        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.resize_token_embeddings(len(tokenizer))

        lora_cfg = build_lora_config_from_args(args)
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        train_tok = preprocess_dataset(train_split, tokenizer, exp_cfg, args.max_input_length, args.max_target_length)
        val_tok = preprocess_dataset(val_split, tokenizer, exp_cfg, args.max_input_length, args.max_target_length)
        logger.info(f"Train samples: {len(train_tok)} | Val samples: {len(val_tok)}")

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

        eval_strategy = "no" if args.skip_eval else "epoch"
        save_strategy = "no" if args.skip_eval else "epoch"
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            warmup_steps=args.warmup_steps,
            weight_decay=0.01,
            logging_steps=50,
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            save_total_limit=1,
            load_best_model_at_end=not args.skip_eval,
            metric_for_best_model="eval_loss",
            predict_with_generate=False,
            fp16=args.fp16,
            report_to="none",
            dataloader_pin_memory=False,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            processing_class=tokenizer,
            data_collator=data_collator,
        )

    # ── Qwen (causal LM) branch ─────────────────────────────────────
    else:
        logger.info(f"Loading causal LM: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if exp_cfg["use_length_tokens"]:
            num_added = tokenizer.add_tokens(SPECIAL_TOKENS)
            logger.info(f"Added {num_added} special tokens")

        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
        model.resize_token_embeddings(len(tokenizer))

        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            model.config.use_cache = False

        lora_cfg = build_causal_lora_config_from_args(args)
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        model = model.to(device)

        train_tok = preprocess_dataset_causal(train_split, tokenizer, exp_cfg, args.max_input_length, args.max_target_length)
        val_tok = preprocess_dataset_causal(val_split, tokenizer, exp_cfg, args.max_input_length, args.max_target_length)
        logger.info(f"Train samples: {len(train_tok)} | Val samples: {len(val_tok)}")

        data_collator = CausalLMDataCollator(pad_token_id=tokenizer.pad_token_id)

        eval_strategy = "no" if args.skip_eval else "epoch"
        save_strategy = "no" if args.skip_eval else "epoch"
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_steps=args.warmup_steps,
            weight_decay=0.01,
            logging_steps=50,
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            save_total_limit=1,
            load_best_model_at_end=not args.skip_eval,
            metric_for_best_model="eval_loss",
            fp16=args.fp16,
            gradient_checkpointing=args.gradient_checkpointing,
            report_to="none",
            dataloader_pin_memory=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            processing_class=tokenizer,
            data_collator=data_collator,
        )

    # ── Train (shared) ───────────────────────────────────────────────
    # Resolve --resume_from_checkpoint: "auto" → True so Trainer finds the latest checkpoint itself
    resume = args.resume_from_checkpoint
    if resume and resume.lower() == "auto":
        resume = True
        logger.info("Auto-resuming from latest checkpoint in output_dir")
    elif resume:
        logger.info(f"Resuming from checkpoint: {resume}")

    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model()

    # Save metrics
    metrics = train_result.metrics
    metrics["experiment"] = args.exp
    metrics["model_name"] = model_name
    metrics["multitask"] = exp_cfg["multitask"]
    metrics["length_tokens"] = exp_cfg["use_length_tokens"]
    metrics["train_samples"] = len(train_tok)
    metrics["val_samples"] = len(val_tok)

    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    logger.info(f"Training metrics saved to {metrics_path}")
    logger.info(f"Final train loss: {metrics.get('train_loss', 'N/A'):.4f}")


def build_lora_config_from_args(args):
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )


def build_causal_lora_config_from_args(args):
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=QWEN_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


if __name__ == "__main__":
    main()
