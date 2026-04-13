---
name: experiment-results-flan-t5
description: FLAN-T5 final v2 results on DialogSum (Exp0/Exp1/Exp1_multi), 10 epochs, lr=5e-5, n=1500 test
type: project
---

## FLAN-T5-base 实验结果（v2 最终版, 2026-04-08）

### 训练环境
- 设备：Mac M4 24GB, MPS
- 模型：google/flan-t5-base (220M), LoRA r=16, alpha=32, target: q/v
- 训练：10 epochs, batch_size=8, lr=5e-5, 自然语言长度指令

### ROUGE 对比（Test n=1500, full test set）

| 实验 | ROUGE-1 | ROUGE-2 | ROUGE-L | Train Loss |
|------|:---:|:---:|:---:|:---:|
| Exp0 Baseline | 30.39 | 11.30 | 25.74 | 14.55 |
| Exp1 Length Control | 30.94 | 11.67 | 26.02 | 14.57 |
| Exp1 Multi-Task | 30.94 | 11.68 | **26.05** | **4.42** |

### Length Control（Exp1/Exp1_multi, n=1500）
- Length Accuracy: 47.8%
- SHORT: 76.5%, MEDIUM: 28.3%, LONG: 19.7%
- Length MAE: 31.87

### 关键结论
1. 长度控制不损害 ROUGE 质量（ROUGE-L: 26.02/26.05 vs baseline 25.74）
2. 多任务学习显著降低 train loss（4.42 vs 14.57, −70%）
3. SHORT 桶控制最好（76.5%），LONG 最难（19.7%，仅占训练集 11%）
4. v2 比 v1 大幅提升：ROUGE-L 从 18.24 提升到 26.05，归功于指令前缀 + lr 5e-5 + 10 epochs

**Why:** v1（3 epochs, lr=5e-4）模型未充分收敛。v2 增加训练量后 ROUGE 提升 43%。

**How to apply:** 当前结果为最终版。对比数据见 `results/metrics/eval_results_*_v2.json` 和 `results/metrics/final_results_v2.json`。
