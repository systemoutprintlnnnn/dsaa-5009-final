#!/usr/bin/env python3
"""Generate comparison figures and final results table.

Reads eval_results_*.json files and produces:
  - results/metrics/final_results_v2.json  (unified summary)
  - results/figures/rouge_comparison.png
  - results/figures/length_accuracy.png
  - results/figures/cross_model.png

Usage:
  PYTHONPATH=. .venv/bin/python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


# ── Load available result files ──────────────────────────────────────

def load(name: str) -> dict | None:
    p = METRICS / f"eval_results_{name}.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


flan_exp0   = load("exp0_v2")          # full 1500-sample re-eval
flan_exp1   = load("exp1_v2")          # full 1500-sample re-eval
flan_multi  = load("exp1_multi_v2")    # full 1500-sample re-eval
qwen_exp0   = load("exp0_qwen")
qwen_exp1   = load("exp1_qwen")
qwen_multi  = load("exp1_multi_qwen")

# Check which results are available
available = {
    "flan_exp0":  flan_exp0  is not None,
    "flan_exp1":  flan_exp1  is not None,
    "flan_multi": flan_multi is not None,
    "qwen_exp0":  qwen_exp0  is not None,
    "qwen_exp1":  qwen_exp1  is not None,
    "qwen_multi": qwen_multi is not None,
}
print("Available results:", {k: v for k, v in available.items() if v})
missing = [k for k, v in available.items() if not v]
if missing:
    print("Missing results:", missing)


def rouge(r: dict | None, key: str) -> float | None:
    if r is None:
        return None
    return r.get("rouge", {}).get(key)


def len_acc(r: dict | None) -> float | None:
    if r is None:
        return None
    v = r.get("length_accuracy")
    return v * 100 if v is not None else None


# ── Build unified summary ─────────────────────────────────────────────

summary = {
    "generated": str(Path(__file__).name),
    "flan_t5_base": {
        "model": "google/flan-t5-base (220M)",
        "train_samples": 12460,
        "train_epochs": 10,
        "note": "Full dataset, full training",
        "exp0_baseline":    {"n": flan_exp0["num_samples"]  if flan_exp0  else None, "rouge1": rouge(flan_exp0, "rouge1"),  "rouge2": rouge(flan_exp0, "rouge2"),  "rougeL": rouge(flan_exp0, "rougeL")},
        "exp1_length_ctrl": {"n": flan_exp1["num_samples"]  if flan_exp1  else None, "rouge1": rouge(flan_exp1, "rouge1"),  "rouge2": rouge(flan_exp1, "rouge2"),  "rougeL": rouge(flan_exp1, "rougeL"),  "length_acc": len_acc(flan_exp1)},
        "exp1_multi_task":  {"n": flan_multi["num_samples"] if flan_multi else None, "rouge1": rouge(flan_multi, "rouge1"), "rouge2": rouge(flan_multi, "rouge2"), "rougeL": rouge(flan_multi, "rougeL"), "length_acc": len_acc(flan_multi)},
    },
    "qwen3_5_0_8b": {
        "model": "Qwen/Qwen3.5-0.8B (753M)",
        "train_samples": 300,
        "train_epochs": 3,
        "note": "Reduced dataset due to MPS linear-attention fallback (~8s/step)",
        "exp0_baseline":    {"n": qwen_exp0["num_samples"]  if qwen_exp0  else None, "rouge1": rouge(qwen_exp0, "rouge1"),  "rouge2": rouge(qwen_exp0, "rouge2"),  "rougeL": rouge(qwen_exp0, "rougeL")},
        "exp1_length_ctrl": {"n": qwen_exp1["num_samples"]  if qwen_exp1  else None, "rouge1": rouge(qwen_exp1, "rouge1"),  "rouge2": rouge(qwen_exp1, "rouge2"),  "rougeL": rouge(qwen_exp1, "rougeL"),  "length_acc": len_acc(qwen_exp1)},
        "exp1_multi_task":  {"n": qwen_multi["num_samples"] if qwen_multi else None, "rouge1": rouge(qwen_multi, "rouge1"), "rouge2": rouge(qwen_multi, "rouge2"), "rougeL": rouge(qwen_multi, "rougeL"), "length_acc": len_acc(qwen_multi)},
    },
}

out_path = METRICS / "final_results_v2.json"
out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"Summary saved to {out_path}")


# ── Figures ───────────────────────────────────────────────────────────

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available — skipping figures")

if HAS_MPL:
    plt.rcParams.update({"figure.dpi": 150, "font.size": 11})

    # ── Figure 1: FLAN-T5 ROUGE comparison ───────────────────────────
    metrics_order = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    flan_data = {
        "Baseline (Exp0)":     [rouge(flan_exp0, "rouge1"), rouge(flan_exp0, "rouge2"), rouge(flan_exp0, "rougeL")],
        "Length Ctrl (Exp1)":  [rouge(flan_exp1, "rouge1"), rouge(flan_exp1, "rouge2"), rouge(flan_exp1, "rougeL")],
        "Multi-Task (Exp1M)":  [rouge(flan_multi, "rouge1"), rouge(flan_multi, "rouge2"), rouge(flan_multi, "rougeL")],
    }

    if any(v[0] is not None for v in flan_data.values()):
        x = np.arange(len(metrics_order))
        width = 0.25
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#4c72b0", "#55a868", "#c44e52"]
        for i, (label, vals) in enumerate(flan_data.items()):
            safe = [v if v is not None else 0 for v in vals]
            bars = ax.bar(x + i * width, safe, width, label=label, color=colors[i])
            for bar, v in zip(bars, vals):
                if v is not None:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                            f"{v:.1f}", ha="center", va="bottom", fontsize=8)

        ax.set_xlabel("Metric")
        ax.set_ylabel("Score")
        ax.set_title("FLAN-T5-base: ROUGE Scores on DialogSum Test Set\n(n=1500, LoRA r=16, 10 epochs)")
        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics_order)
        ax.legend()
        ax.set_ylim(0, 40)
        fig.tight_layout()
        fig.savefig(FIGURES / "flan_rouge_comparison.png")
        plt.close(fig)
        print("Saved flan_rouge_comparison.png")

    # ── Figure 2: Length accuracy per bucket ─────────────────────────
    if flan_exp1 is not None and "length_accuracy_short" in flan_exp1:
        buckets = ["SHORT", "MEDIUM", "LONG", "Overall"]
        exp1_acc = [
            flan_exp1.get("length_accuracy_short", 0) * 100,
            flan_exp1.get("length_accuracy_medium", 0) * 100,
            flan_exp1.get("length_accuracy_long", 0) * 100,
            flan_exp1.get("length_accuracy", 0) * 100,
        ]
        multi_acc = [
            flan_multi.get("length_accuracy_short", 0) * 100 if flan_multi else 0,
            flan_multi.get("length_accuracy_medium", 0) * 100 if flan_multi else 0,
            flan_multi.get("length_accuracy_long", 0) * 100 if flan_multi else 0,
            flan_multi.get("length_accuracy", 0) * 100 if flan_multi else 0,
        ]

        x = np.arange(len(buckets))
        width = 0.35
        fig, ax = plt.subplots(figsize=(7, 5))
        bars1 = ax.bar(x - width / 2, exp1_acc, width, label="Exp1 (Single-Task)", color="#55a868")
        bars2 = ax.bar(x + width / 2, multi_acc, width, label="Exp1M (Multi-Task)", color="#c44e52")
        for bar, v in zip(list(bars1) + list(bars2), exp1_acc + multi_acc):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

        ax.set_xlabel("Length Bucket")
        ax.set_ylabel("Length Accuracy (%)")
        ax.set_title("FLAN-T5: Length Control Accuracy by Bucket")
        ax.set_xticks(x)
        ax.set_xticklabels(buckets)
        ax.legend()
        ax.set_ylim(0, 100)
        fig.tight_layout()
        fig.savefig(FIGURES / "length_accuracy.png")
        plt.close(fig)
        print("Saved length_accuracy.png")

    # ── Figure 3: Cross-model ROUGE-L comparison ─────────────────────
    if qwen_exp0 is not None:
        labels = ["Baseline (Exp0)", "Length Ctrl (Exp1)", "Multi-Task (Exp1M)"]
        flan_rl = [rouge(flan_exp0, "rougeL"), rouge(flan_exp1, "rougeL"), rouge(flan_multi, "rougeL")]
        qwen_rl = [rouge(qwen_exp0, "rougeL"), rouge(qwen_exp1, "rougeL"), rouge(qwen_multi, "rougeL")]

        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        safe_flan = [v if v is not None else 0 for v in flan_rl]
        safe_qwen = [v if v is not None else 0 for v in qwen_rl]
        bars1 = ax.bar(x - width / 2, safe_flan, width, label="FLAN-T5-base (220M, n=1500)", color="#4c72b0")
        bars2 = ax.bar(x + width / 2, safe_qwen, width, label="Qwen3.5-0.8B (753M, n=500)", color="#dd8452")
        for bar, v in zip(list(bars1) + list(bars2), safe_flan + safe_qwen):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=8)

        ax.set_xlabel("Experiment")
        ax.set_ylabel("ROUGE-L")
        ax.set_title("Cross-Model ROUGE-L Comparison\n(FLAN-T5: 12460 train samples / Qwen: 300 train samples)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / "cross_model_rougeL.png")
        plt.close(fig)
        print("Saved cross_model_rougeL.png")


# ── Print summary table ───────────────────────────────────────────────
print("\n" + "=" * 65)
print("RESULTS SUMMARY")
print("=" * 65)
print(f"{'Experiment':<30} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} {'Len-Acc':>8}")
print("-" * 65)

rows = [
    ("FLAN-T5 Exp0 Baseline",    flan_exp0,  False),
    ("FLAN-T5 Exp1 Length Ctrl", flan_exp1,  True),
    ("FLAN-T5 Exp1M Multi-Task", flan_multi, True),
    ("Qwen Exp0 Baseline",       qwen_exp0,  False),
    ("Qwen Exp1 Length Ctrl",    qwen_exp1,  True),
    ("Qwen Exp1M Multi-Task",    qwen_multi, True),
]

for name, r, has_len in rows:
    r1 = f"{rouge(r,'rouge1'):.2f}" if rouge(r,'rouge1') else "—"
    r2 = f"{rouge(r,'rouge2'):.2f}" if rouge(r,'rouge2') else "—"
    rl = f"{rouge(r,'rougeL'):.2f}" if rouge(r,'rougeL') else "—"
    la = f"{len_acc(r):.1f}%" if (has_len and len_acc(r) is not None) else "—"
    print(f"{name:<30} {r1:>8} {r2:>8} {rl:>8} {la:>8}")

print("=" * 65)
