# CHECKPOINTS

> Project: DSAA-5009 Final Project - Multi-Task Learning for Length-Controllable Dialogue Summarization
> Purpose: stage-by-stage acceptance mechanism
> Last updated: 2026-04-08

---

## Status Legend
- 🟢 **PASS**: verified, safe to move on
- 🟡 **PARTIAL**: usable but has risk / needs follow-up
- 🔴 **FAIL**: blocked, do not proceed
- ⚪ **PENDING**: not started yet

---

## CP-01 项目骨架初始化
- **Status**: 🟢 PASS
- **Goal**: 完成项目基础目录、任务追踪文件、初始骨架
- **Verification**:
  - [x] `src/`, `scripts/`, `results/` 目录存在
  - [x] `TASKS.md` 存在
  - [x] 基础 Python package 结构建立
- **Result**: 基础骨架已建立

---

## CP-02 数据分析与长度分桶验证
- **Status**: 🟢 PASS
- **Goal**: 验证 DialogSum 数据集可读取，长度分布合理，分桶阈值可用
- **Verification**:
  - [x] `scripts/analyze_data.py` 可运行
  - [x] 输出 SHORT/MEDIUM/LONG 分桶比例
  - [x] 生成 `results/metrics/data_stats.json`
  - [x] 生成 `results/metrics/length_distribution.png`
- **Artifacts**: `scripts/analyze_data.py`, `results/metrics/data_stats.json`
- **Result**: 数据读取、长度统计与分桶验证通过

---

## CP-03 多任务数据管线验证
- **Status**: 🟢 PASS
- **Goal**: 验证 summary + topic 的多任务数据构建逻辑正确
- **Verification**:
  - [x] 能正确生成摘要任务和主题任务样本
  - [x] 长度 token 能正确注入
  - [x] 生成 `results/metrics/multitask_samples.json`
- **Artifacts**: `src/data/preprocessing.py`, `scripts/check_multitask_data.py`
- **Result**: 多任务样本构造通过

---

## CP-04 模型加载验证
- **Status**: 🟢 PASS
- **Goal**: 验证模型、tokenizer、special tokens、LoRA 配置可正常加载
- **Verification**:
  - [x] FLAN-T5 正常加载
  - [x] tokenizer vocab 成功扩展（5 special tokens）
  - [x] LoRA 配置成功注入，trainable%: 0.7877%
  - [x] 生成 `results/metrics/model_check_flan.json`
- **Artifacts**: `src/models/load_model.py`, `scripts/check_model_loading.py`
- **Result**: 模型、tokenizer、LoRA 注入验证通过

---

## CP-05 单步训练验证
- **Status**: 🟢 PASS
- **Goal**: 验证训练流程至少能跑通 1 batch
- **Verification**:
  - [x] `scripts/check_training_step.py` 成功
  - [x] loss=459.45，有限值、非 NaN
  - [x] forward + backward 正常
  - [x] 生成 `results/metrics/training_smoke_test.json`
- **Artifacts**: `src/training/trainer.py`, `scripts/check_training_step.py`
- **Result**: Smoke test 通过（MPS 设备）

---

## CP-06 Baseline 训练完成
- **Status**: 🟢 PASS
- **Goal**: 完成 Exp0 baseline 训练并产出可评测模型
- **Verification**:
  - [x] `scripts/run_training.py --exp exp0` 成功
  - [x] checkpoint 文件存在
  - [x] 训练日志完整
- **Artifacts**: `results/models/exp0_v2/`, `results/models/exp0_v2/training_metrics.json`
- **Result**: Exp0 v2 完成，10 epochs，lr=5e-5，train_loss=14.55，ROUGE-L=23.13

---

## CP-07 评测管线验证
- **Status**: 🟢 PASS
- **Goal**: 验证 ROUGE + Length Accuracy 能正确计算
- **Verification**:
  - [x] `scripts/run_evaluation.py` 成功
  - [x] ROUGE 成功输出
  - [x] Length Accuracy 成功输出
  - [x] 生成 `results/metrics/eval_results_exp0.json`
- **Artifacts**: `src/evaluation/rouge.py`, `src/evaluation/length_metrics.py`, `scripts/run_evaluation.py`
- **Result**: 评测管线验证通过

---

## CP-08 长度控制实验完成
- **Status**: 🟢 PASS
- **Goal**: 完成 Exp1，验证长度控制有效
- **Verification**:
  - [x] Exp1 v2 训练完成（10 epochs，自然语言长度指令）
  - [x] SHORT/MEDIUM/LONG 三类摘要生成成功
  - [x] Length Accuracy = 54.5%，SHORT = 80.6%
  - [x] ROUGE 不低于 baseline（23.45 vs 23.13）
- **Artifacts**: `results/models/exp1_v2/`, `results/metrics/eval_results_exp1.json`
- **Result**: 长度控制有效，Length Acc 从旧版 28.7% 提升到 54.5%

---

## CP-09 多任务学习实验完成
- **Status**: 🟢 PASS
- **Goal**: 完成多任务学习实验并验证是否提升摘要质量
- **Verification**:
  - [x] 多任务训练成功（24,920 samples × 10 epochs）
  - [x] train_loss = 4.42，远低于单任务的 14.57
  - [x] ROUGE-L = 23.48，略优于单任务（23.45）和 baseline（23.13）
  - [x] Length Accuracy = 54.5%，与单任务持平
- **Artifacts**: `results/models/exp1_multi_v2/`, `results/metrics/eval_results_exp1_multi.json`
- **Result**: 多任务学习显著降低训练 loss，ROUGE 略有提升

---

## CP-10 结果汇总与报告材料完成
- **Status**: 🟢 PASS
- **Goal**: 完成对比表、关键结论整理
- **Verification**:
  - [x] 三组实验对比表生成
  - [x] final_results.json 更新
  - [x] 关键结论整理完成
- **Artifacts**: `results/metrics/final_results.json`, 本文件
- **Result**: 全部实验结果已汇总

---

## 实验结果总表（FLAN-T5-base, 10 epochs, lr=5e-5, LoRA r=16, test n=1500）

| 指标 | Exp0 Baseline | Exp1 Length Control | Exp1 Multi-Task |
|------|:---:|:---:|:---:|
| ROUGE-1 | 30.39 | **30.94** | **30.94** |
| ROUGE-2 | 11.30 | 11.67 | **11.68** |
| ROUGE-L | 25.74 | 26.02 | **26.05** |
| ROUGE-Lsum | 25.74 | 26.01 | **26.05** |
| Train Loss | 14.55 | 14.57 | **4.42** |
| Length Accuracy | — | **47.8%** | **47.8%** |
| Length Acc (SHORT) | — | **76.5%** | **76.5%** |
| Length Acc (MEDIUM) | — | 28.3% | 28.3% |
| Length Acc (LONG) | — | 19.7% | 19.7% |
| Length MAE | — | **31.87** | **31.87** |

## 关键结论（FLAN-T5, n=1500 full test）

1. **长度控制有效**：自然语言长度指令使 Length Accuracy 达到 47.8%，SHORT 桶达 76.5%
2. **长度控制不损害质量**：Exp1/Exp1_multi 的 ROUGE 均略高于 baseline（ROUGE-L: 26.02/26.05 vs 25.74）
3. **多任务学习显著降低训练 loss**：Exp1_multi 的 train_loss（4.42）比单任务（14.57）低 70%
4. **v2 比 v1 大幅提升**：ROUGE-L 从 18.24（v1）提升到 26.05（v2），主要归功于添加指令前缀和降低学习率

---

## CP-11 Qwen Baseline 训练（exp0_qwen）
- **Status**: 🟢 PASS
- **Goal**: Qwen/Qwen3.5-0.8B baseline 训练完成，无长度控制
- **Verification**:
  - [x] `scripts/run_all_qwen.sh` 成功完成（含 --skip_eval 修复 MPS 碎片化问题）
  - [x] checkpoint 存在于 `results/models/exp0_qwen/`
  - [x] `training_metrics.json` 记录 train_loss
- **Artifacts**: `results/models/exp0_qwen/`
- **Result**: 300 samples × 5 epochs, ROUGE-L=34.31, ROUGE-1=42.21（n=500 test）

---

## CP-12 Qwen 长度控制训练（exp1_qwen）
- **Status**: 🟢 PASS
- **Goal**: Qwen 长度控制实验完成
- **Verification**:
  - [x] `scripts/run_all_qwen.sh` 成功完成
  - [x] checkpoint 存在于 `results/models/exp1_qwen/`
- **Artifacts**: `results/models/exp1_qwen/`
- **Result**: ROUGE-L=33.41, Length Accuracy=68.2%（SHORT=65.5%, MEDIUM=74.7%, LONG=26.7%）

---

## CP-13 Qwen 多任务训练（exp1_multi_qwen）
- **Status**: 🟢 PASS
- **Goal**: Qwen 多任务实验（摘要+主题）完成
- **Verification**:
  - [x] `scripts/run_all_qwen.sh` 成功完成
  - [x] checkpoint 存在于 `results/models/exp1_multi_qwen/`
- **Artifacts**: `results/models/exp1_multi_qwen/`
- **Result**: ROUGE-L=34.29, Length Accuracy=74.2%（SHORT=68.0%, MEDIUM=82.1%, LONG=43.3%）
- **关键发现**: 多任务使 Length Acc 从 68.2% 提升至 74.2%（+6%），ROUGE-L 恢复至 34.29（≈baseline 34.31）

---

## CP-14 Qwen 评测完成
- **Status**: 🟢 PASS
- **Goal**: 三组 Qwen 实验评测指标全部产出
- **Verification**:
  - [x] `eval_results_exp0_qwen.json` 存在（n=500）
  - [x] `eval_results_exp1_qwen.json` 存在（n=500）
  - [x] `eval_results_exp1_multi_qwen.json` 存在（n=500）
  - [x] ROUGE + Length Accuracy 数值合理
- **Artifacts**: `results/metrics/eval_results_exp*_qwen.json`

---

## CP-15 FLAN-T5 全量重评（1500 samples）
- **Status**: 🟢 PASS
- **Goal**: 用完整 test set 重新评测三组 FLAN-T5 实验
- **Verification**:
  - [x] `eval_results_exp0_v2.json`（n=1500）: ROUGE-L=25.74
  - [x] `eval_results_exp1_v2.json`（n=1500）: ROUGE-L=26.02, Length Acc=47.8%
  - [x] `eval_results_exp1_multi_v2.json`（n=1500）: ROUGE-L=26.05, Length Acc=47.8%
- **Artifacts**: `results/metrics/eval_results_*_v2.json`
