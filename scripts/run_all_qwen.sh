#!/usr/bin/env bash
# Run all Qwen experiments + evaluations sequentially.
# Uses .venv/bin/python to ensure correct environment.
# Usage: bash scripts/run_all_qwen.sh 2>&1 | tee logs/run_all_qwen.log

set -e
PYTHON=".venv/bin/python"
COMMON="--batch_size 4 --max_input_length 256 --max_target_length 64 --epochs 5 --max_train_samples 300 --lr 5e-5 --warmup_steps 30 --skip_eval"

echo "============================================================"
echo "STEP 1/6  Train exp0_qwen (baseline, no length control)"
echo "============================================================"
PYTHONPATH=. $PYTHON scripts/run_training.py --exp exp0_qwen $COMMON

echo "============================================================"
echo "STEP 2/6  Train exp1_qwen (length-controllable)"
echo "============================================================"
PYTHONPATH=. $PYTHON scripts/run_training.py --exp exp1_qwen $COMMON

echo "============================================================"
echo "STEP 3/6  Train exp1_multi_qwen (multi-task)"
echo "============================================================"
PYTHONPATH=. $PYTHON scripts/run_training.py --exp exp1_multi_qwen $COMMON

echo "============================================================"
echo "STEP 4/6  Evaluate FLAN-T5 exp1_v2 (1500 test samples)"
echo "============================================================"
PYTHONPATH=. $PYTHON scripts/run_evaluation.py --exp exp1 --model_dir results/models/exp1_v2 --split test

echo "============================================================"
echo "STEP 5/6  Evaluate FLAN-T5 exp1_multi_v2 (1500 test samples)"
echo "============================================================"
PYTHONPATH=. $PYTHON scripts/run_evaluation.py --exp exp1_multi --model_dir results/models/exp1_multi_v2 --split test

echo "============================================================"
echo "STEP 6/6  Evaluate all Qwen models (500 test samples — training was 300 samples)"
echo "============================================================"
PYTHONPATH=. $PYTHON scripts/run_evaluation.py --exp exp0_qwen --split test --max_samples 500
PYTHONPATH=. $PYTHON scripts/run_evaluation.py --exp exp1_qwen --split test --max_samples 500
PYTHONPATH=. $PYTHON scripts/run_evaluation.py --exp exp1_multi_qwen --split test --max_samples 500

echo "============================================================"
echo "ALL DONE — results in results/metrics/"
echo "============================================================"
