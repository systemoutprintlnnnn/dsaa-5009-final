#!/usr/bin/env python3
"""Evaluation script for dialogue summarization experiments.

Computes ROUGE, Length Accuracy, Length MAE.

Usage:
  PYTHONPATH=. python scripts/run_evaluation.py --exp exp0
  PYTHONPATH=. python scripts/run_evaluation.py --exp exp1
  PYTHONPATH=. python scripts/run_evaluation.py --exp exp1_multi
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.data.preprocessing import BucketConfig, get_length_bucket
from src.evaluation.rouge import compute_rouge
from src.evaluation.length_metrics import (
    BUCKET_RANGES,
    length_accuracy,
    length_mae,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "metrics"

EXPERIMENT_DIRS = {
    "exp0":           ROOT / "results" / "models" / "exp0",
    "exp1":           ROOT / "results" / "models" / "exp1",
    "exp1_multi":     ROOT / "results" / "models" / "exp1_multi",
    "exp0_qwen":      ROOT / "results" / "models" / "exp0_qwen",
    "exp1_qwen":      ROOT / "results" / "models" / "exp1_qwen",
    "exp1_multi_qwen": ROOT / "results" / "models" / "exp1_multi_qwen",
}

MODEL_TYPES = {
    "exp0":            "seq2seq",
    "exp1":            "seq2seq",
    "exp1_multi":      "seq2seq",
    "exp0_qwen":       "causal",
    "exp1_qwen":       "causal",
    "exp1_multi_qwen": "causal",
}

USE_LENGTH_TOKENS = {"exp1", "exp1_multi", "exp1_qwen", "exp1_multi_qwen"}

# Must match training script LENGTH_INSTRUCTIONS
LENGTH_INSTRUCTIONS = {
    "SHORT": "Write a very brief one-sentence summary of the dialogue in 5 to 15 words.",
    "MEDIUM": "Write a short summary of the dialogue in 16 to 35 words.",
    "LONG": "Write a detailed summary of the dialogue in more than 35 words.",
}


def generate_summaries(model, tokenizer, inputs, device, max_input=512, max_target=128):
    """Generate summaries for a list of pre-formatted inputs."""
    model.eval()
    predictions = []
    batch_size = 8

    for i in tqdm(range(0, len(inputs), batch_size), desc="Generating"):
        batch_texts = inputs[i : i + batch_size]
        tokenized = tokenizer(
            batch_texts,
            max_length=max_input,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **tokenized,
                max_new_tokens=max_target,
                num_beams=4,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)

    return predictions


def generate_summaries_causal(model, tokenizer, prompts, device, max_input=512, max_new_tokens=128):
    """Generate summaries using a causal LM; decode only the newly generated tokens."""
    model.eval()
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # left-pad for batch generation

    predictions = []
    batch_size = 4

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[i : i + batch_size]
        tokenized = tokenizer(
            batch,
            max_length=max_input,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        prompt_len = tokenized["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **tokenized,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                num_beams=1,
            )

        new_tokens = output_ids[:, prompt_len:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        predictions.extend([d.strip() for d in decoded])

    tokenizer.padding_side = original_padding_side
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate dialogue summarization model")
    parser.add_argument("--exp", choices=list(EXPERIMENT_DIRS.keys()), required=True,
                        help="Experiment name (e.g. exp1_multi, exp1_multi_qwen)")
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--max_samples", type=int, default=None, help="Limit eval samples")
    parser.add_argument("--model_dir", default=None, help="Override model checkpoint dir")
    args = parser.parse_args()

    # Load model
    model_dir = Path(args.model_dir) if args.model_dir else EXPERIMENT_DIRS[args.exp]
    if not model_dir.exists():
        logger.error(f"Model dir not found: {model_dir}")
        return

    model_type = MODEL_TYPES[args.exp]
    logger.info(f"Loading {model_type} model from {model_dir}")

    # Device
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    if model_type == "causal":
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Load base model → resize embedding to match saved tokenizer → apply LoRA adapter.
        # This avoids the "ignore_mismatched_sizes" error when vocab was resized during training.
        from peft import PeftConfig
        peft_config = PeftConfig.from_pretrained(str(model_dir))
        base_model_name = peft_config.base_model_name_or_path
        logger.info(f"Loading base model: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=torch.bfloat16)
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, str(model_dir))
        model = model.merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))

    model = model.to(device)

    # Load dataset
    ds = load_dataset("knkarthick/dialogsum", split=args.split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    use_len_tokens = args.exp in USE_LENGTH_TOKENS

    # ── Build evaluation inputs (must match training format exactly) ──
    ref_summaries = [row["summary"] for row in ds]
    bucket_cfg = BucketConfig()
    target_buckets = [get_length_bucket(s, bucket_cfg) for s in ref_summaries]

    # Causal LM prompts end with "Summary: " or "Topic: " to trigger generation;
    # seq2seq prompts do not need this suffix (encoder handles it).
    suffix = "\nSummary: " if model_type == "causal" else ""

    if use_len_tokens:
        inputs = []
        for row in ds:
            bucket = get_length_bucket(row["summary"], bucket_cfg)
            length_instr = LENGTH_INSTRUCTIONS[bucket]
            inp = (
                f"Summarize the following dialogue.\n"
                f"Instruction: {length_instr}\n"
                f"Dialogue:\n{row['dialogue']}"
                f"{suffix}"
            )
            inputs.append(inp)
    else:
        inputs = [
            f"Summarize the following dialogue.\nDialogue:\n{row['dialogue']}{suffix}"
            for row in ds
        ]

    # ── Generate predictions ─────────────────────────────────────
    logger.info(f"Generating summaries for {len(inputs)} samples...")
    if model_type == "causal":
        predictions = generate_summaries_causal(model, tokenizer, inputs, device)
    else:
        predictions = generate_summaries(model, tokenizer, inputs, device)

    # ── Compute metrics ──────────────────────────────────────────
    logger.info("Computing ROUGE...")
    rouge_results = compute_rouge(predictions, ref_summaries)

    results = {
        "experiment": args.exp,
        "split": args.split,
        "num_samples": len(predictions),
        "rouge": rouge_results,
    }

    # Length metrics (only meaningful with length tokens)
    if use_len_tokens:
        logger.info("Computing length metrics...")
        len_acc = length_accuracy(predictions, target_buckets)
        len_mae_val = length_mae(predictions, target_buckets)
        results["length_accuracy"] = len_acc
        results["length_mae"] = len_mae_val

        # Per-bucket accuracy
        for bucket in ["SHORT", "MEDIUM", "LONG"]:
            idxs = [i for i, b in enumerate(target_buckets) if b == bucket]
            if idxs:
                bucket_preds = [predictions[i] for i in idxs]
                bucket_targets = [target_buckets[i] for i in idxs]
                results[f"length_accuracy_{bucket.lower()}"] = length_accuracy(bucket_preds, bucket_targets)

    # ── Save ─────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Use model_dir name for output file so custom --model_dir also works
    output_tag = model_dir.name
    output_path = RESULTS_DIR / f"eval_results_{output_tag}.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # Print summary
    print(f"\n{'='*50}")
    print(f"Evaluation: {args.exp} | split={args.split} | n={len(predictions)}")
    print(f"{'='*50}")
    print(f"ROUGE-1:  {rouge_results.get('rouge1', 0):.2f}")
    print(f"ROUGE-2:  {rouge_results.get('rouge2', 0):.2f}")
    print(f"ROUGE-L:  {rouge_results.get('rougeL', 0):.2f}")
    print(f"ROUGE-Lsum: {rouge_results.get('rougeLsum', 0):.2f}")
    if use_len_tokens:
        print(f"Length Accuracy: {results['length_accuracy']*100:.1f}%")
        print(f"Length MAE: {results['length_mae']:.2f}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
