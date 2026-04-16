#!/bin/bash
# Rerun all experiments with corrected settings:
#   - No unused special tokens (removed embedding resize noise)
#   - Fixed multi-task tokenization (per-sample max_length)
#   - Consistent LoRA config (q_proj, v_proj only for Qwen)
#   - More FLAN-T5 epochs (20 instead of 10)
#
# Usage:
#   cd /Users/tjzhou/Desktop/AIWorkspace/homework/dsaa-5009-final
#   bash scripts/rerun_all.sh          # run everything
#   bash scripts/rerun_all.sh flan     # FLAN-T5 only
#   bash scripts/rerun_all.sh qwen     # Qwen only

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOGDIR="$ROOT/logs/rerun"
mkdir -p "$LOGDIR"

SECTION="${1:-all}"

run() {
    local name="$1"
    shift
    echo "============================================================"
    echo "Training: $name"
    echo "============================================================"
    PYTHONPATH=. python scripts/run_training.py "$@" 2>&1 | tee "$LOGDIR/${name}.log"
    echo ""
}

eval_model() {
    local exp="$1"
    local model_dir="$2"
    echo "============================================================"
    echo "Evaluating: $exp from $model_dir"
    echo "============================================================"
    PYTHONPATH=. python scripts/run_evaluation.py --exp "$exp" --model_dir "$model_dir" 2>&1 | tee "$LOGDIR/eval_${exp}.log"
    echo ""
}

# ── FLAN-T5 (seq2seq) ────────────────────────────────────────────────
# Key changes vs. previous run:
#   - 20 epochs (was 10, training loss was still ~14.5 at epoch 10)
#   - No special token embedding noise
#   - Fixed multi-task tokenization
if [[ "$SECTION" == "all" || "$SECTION" == "flan" ]]; then

    run "exp0_v3" --exp exp0 --epochs 20 --batch_size 8 --lr 5e-5 --skip_eval
    eval_model "exp0" "results/models/exp0"

    run "exp1_v3" --exp exp1 --epochs 20 --batch_size 8 --lr 5e-5 --skip_eval
    eval_model "exp1" "results/models/exp1"

    run "exp1_multi_v3" --exp exp1_multi --epochs 20 --batch_size 8 --lr 5e-5 --skip_eval
    eval_model "exp1_multi" "results/models/exp1_multi"

fi

# ── Qwen full dataset (causal LM) ────────────────────────────────────
# Key changes vs. previous full run:
#   - LoRA target: q_proj, v_proj ONLY (was 6 modules → overfitting)
#   - No special token embedding noise
#   - batch_size 8 with grad_accum 2 (effective 16)
#   - 5 epochs (sufficient for full 12,460 dataset)
if [[ "$SECTION" == "all" || "$SECTION" == "qwen" ]]; then

    run "exp0_qwen_full_v2" --exp exp0_qwen_full --epochs 5 --batch_size 8 --grad_accum 2 --lr 5e-5 --skip_eval --gradient_checkpointing
    eval_model "exp0_qwen_full" "models/exp0_qwen_full"

    run "exp1_qwen_full_v2" --exp exp1_qwen_full --epochs 5 --batch_size 8 --grad_accum 2 --lr 5e-5 --skip_eval --gradient_checkpointing
    eval_model "exp1_qwen_full" "models/exp1_qwen_full"

    run "exp1_multi_qwen_full_v2" --exp exp1_multi_qwen_full --epochs 5 --batch_size 8 --grad_accum 2 --lr 5e-5 --skip_eval --gradient_checkpointing
    eval_model "exp1_multi_qwen_full" "models/exp1_multi_qwen_full"

fi

echo "============================================================"
echo "All experiments complete. Results:"
echo "============================================================"
for f in results/metrics/eval_results_*.json; do
    [ -f "$f" ] && echo "--- $(basename "$f") ---" && python -c "
import json, sys
d = json.load(open('$f'))
r = d.get('rouge', {})
print(f\"  ROUGE-1={r.get('rouge1',0):.2f}  ROUGE-L={r.get('rougeL',0):.2f}\")
if 'length_accuracy' in d:
    print(f\"  Length Acc={d['length_accuracy']*100:.1f}%  SHORT={d.get('length_accuracy_short',0)*100:.1f}%  MEDIUM={d.get('length_accuracy_medium',0)*100:.1f}%  LONG={d.get('length_accuracy_long',0)*100:.1f}%\")
"
done
