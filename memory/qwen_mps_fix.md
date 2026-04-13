---
name: Qwen MPS Training Issues and Fix
description: Root cause and fix for Qwen3.5-0.8B training speed degradation on Apple Silicon MPS
type: project
---

# Qwen3.5-0.8B on Apple M4 MPS — Training Issues

## Root Cause: Linear Attention Fallback
Qwen3.5-0.8B uses a hybrid architecture with linear attention layers that require
`flash-linear-attention` and `causal-conv1d` libraries (CUDA-only). Without them,
falls back to slow torch implementation. Warning at load time:
> "The fast path is not available because one of the required library is not installed."

**Speed impact**: ~4-6s/step with batch_size=4, max_input=256 (manageable)

## Root Cause: MPS Memory Fragmentation After Checkpoint Save
When `eval_strategy="epoch"` and `save_strategy="epoch"` are used together:
1. Training runs at 5-10s/step (epoch 1 normal)
2. After epoch 1: eval runs, checkpoint saved (includes full embedding: 248047×1024 tensor)
3. **Epoch 2 speed drops to 20-50s/step** due to MPS allocator fragmentation after large save

## Fix: --skip_eval Flag
Added `--skip_eval` CLI flag to `scripts/run_training.py`. When set:
- `eval_strategy="no"` (no mid-training eval)
- `save_strategy="no"` (no mid-training checkpoint saves)
- Final model saved via `trainer.save_model()` at end

**Result**: Consistent ~4s/step across ALL epochs (no degradation after epoch 1)

## Confirmed Working Parameters (2026-04-11)
```
.venv/bin/python scripts/run_training.py \
  --exp exp0_qwen \
  --batch_size 4 --max_input_length 256 --max_target_length 64 \
  --epochs 5 --max_train_samples 300 \
  --lr 5e-5 --warmup_steps 30 --skip_eval
```
Speed: 4-5s/step × 375 steps = ~25 min per experiment

## Training Limitations (Hardware Constraint)
Full dataset (12460 samples) at 4s/step = ~3115 steps/epoch × 5 = 15575 steps × 4s ≈ 17 hours.
**Not feasible on local MPS.**

Decision: Train on 300 samples, 5 epochs as proof-of-concept cross-model comparison.
Note this limitation in paper as "hardware constraint (MPS linear-attention fallback)".

## Python Environment
Use `.venv/bin/python` (NOT `python` or `python3` — system Python lacks ML libs)
`PYTHONPATH=.` must be set for src/ imports.

**Why**: `.venv/` created by uv in project root, contains PyTorch 2.11.0 + transformers.
