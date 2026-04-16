#!/bin/bash
# Rerun Qwen full-dataset experiments with corrected settings:
#   - LoRA target: q_proj, v_proj ONLY (was 6 modules → overfitting)
#   - No unused special tokens (removed embedding resize noise)
#   - Fixed multi-task tokenization (per-sample max_length)
#
# Usage (run from project root):
#   bash scripts/rerun_qwen.sh          # run all 3 experiments
#   bash scripts/rerun_qwen.sh exp0     # run only exp0 baseline
#   bash scripts/rerun_qwen.sh eval     # evaluate all 3 (skip training)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
LOGDIR="$ROOT/logs/rerun"
mkdir -p "$LOGDIR"

ONLY="${1:-all}"

run() {
    local name="$1"
    shift
    echo "============================================================"
    echo "[$(date '+%H:%M:%S')] Training: $name"
    echo "============================================================"
    PYTHONPATH=. "$PY" scripts/run_training.py "$@" 2>&1 | tee "$LOGDIR/${name}.log"
    echo ""
}

eval_model() {
    local exp="$1"
    local model_dir="$2"
    echo "============================================================"
    echo "[$(date '+%H:%M:%S')] Evaluating: $exp"
    echo "============================================================"
    PYTHONPATH=. "$PY" scripts/run_evaluation.py --exp "$exp" --model_dir "$model_dir" --max_samples 500 2>&1 | tee "$LOGDIR/eval_${exp}.log"
    echo ""
}

if [[ "$ONLY" == "eval" ]]; then
    eval_model "exp0_qwen_full"       "models/exp0_qwen_full"
    eval_model "exp1_qwen_full"       "models/exp1_qwen_full"
    eval_model "exp1_multi_qwen_full" "models/exp1_multi_qwen_full"
    echo "Done evaluating."
    exit 0
fi

# ── Train ─────────────────────────────────────────────────────────────
# batch_size=8, grad_accum=2 → effective batch 16
# 5 epochs × 12,460 samples / 8 = ~7,788 optimizer steps
# gradient_checkpointing saves memory on MPS

if [[ "$ONLY" == "all" || "$ONLY" == "exp0" ]]; then
    run "exp0_qwen_full_v2" \
        --exp exp0_qwen_full \
        --epochs 5 --batch_size 8 --grad_accum 2 \
        --lr 5e-5 --skip_eval --gradient_checkpointing
    eval_model "exp0_qwen_full" "models/exp0_qwen_full"
fi

if [[ "$ONLY" == "all" || "$ONLY" == "exp1" ]]; then
    run "exp1_qwen_full_v2" \
        --exp exp1_qwen_full \
        --epochs 5 --batch_size 8 --grad_accum 2 \
        --lr 5e-5 --skip_eval --gradient_checkpointing
    eval_model "exp1_qwen_full" "models/exp1_qwen_full"
fi

if [[ "$ONLY" == "all" || "$ONLY" == "exp1_multi" ]]; then
    run "exp1_multi_qwen_full_v2" \
        --exp exp1_multi_qwen_full \
        --epochs 5 --batch_size 8 --grad_accum 2 \
        --lr 5e-5 --skip_eval --gradient_checkpointing
    eval_model "exp1_multi_qwen_full" "models/exp1_multi_qwen_full"
fi

echo "============================================================"
echo "All done. Results:"
echo "============================================================"
for d in models/exp0_qwen_full models/exp1_qwen_full models/exp1_multi_qwen_full; do
    f="$d/eval_results.json"
    [ -f "$f" ] || continue
    echo "--- $(basename "$d") ---"
    "$PY" -c "
import json
d = json.load(open('$f'))
print(f\"  ROUGE-1={d['rouge1']:.2f}  ROUGE-2={d['rouge2']:.2f}  ROUGE-L={d['rougeL']:.2f}\")
if 'length_acc' in d:
    print(f\"  Len Acc={d['length_acc']*100:.1f}%  S={d.get('len_acc_short',0)*100:.1f}%  M={d.get('len_acc_medium',0)*100:.1f}%  L={d.get('len_acc_long',0)*100:.1f}%\")
"
done
