# TASKS

> Project: DSAA-5009 Final Project - Multi-Task Learning for Length-Controllable Dialogue Summarization
> Last updated: 2026-04-08

---

## Done

### Phase 0 - 项目初始化
- [x] 创建 `README.md`
- [x] 创建 `requirements.txt`
- [x] 创建 `TASKS.md`

### Phase 1 - 数据部分
- [x] 下载并读取 DialogSum 数据集
- [x] 统计 summary 长度分布
- [x] 验证 SHORT / MEDIUM / LONG 分桶是否合理
- [x] 编写长度分桶函数
- [x] 编写多任务数据构建逻辑（summary + topic）
- [x] 编写 tokenizer 预处理代码
- [x] 保存数据分析结果到 `results/metrics/`

### Phase 2 - 模型与训练框架
- [x] 编写模型加载模块 `src/models/load_model.py`
- [x] 编写特殊 token 注入逻辑
- [x] 编写 LoRA 配置逻辑
- [x] 编写 Trainer 封装 `src/training/trainer.py`
- [x] 创建完整训练脚本 `scripts/run_training.py`（支持 exp0/exp1/exp1_multi）
- [x] 添加 FLAN-T5 指令前缀（关键修复）
- [x] 改用自然语言长度指令替代 special tokens
- [x] 修复超参数：lr 5e-5，默认 10 epochs
- [x] 添加 checkpoint 恢复功能

### Phase 3 - 评测框架
- [x] 编写 ROUGE 评测模块 `src/evaluation/rouge.py`
- [x] 编写 Length Accuracy / Length MAE 指标 `src/evaluation/length_metrics.py`
- [x] 编写统一评测脚本 `scripts/run_evaluation.py`

### Phase 4 - 实验执行（v2 最终版）
- [x] Exp0 baseline 训练（10 epochs，指令前缀，lr=5e-5）→ train_loss=14.55
- [x] Exp0 评测 → ROUGE-L=23.13
- [x] Exp1 长度控制训练（自然语言长度指令）→ train_loss=14.57
- [x] Exp1 评测 → ROUGE-L=23.45，Length Acc=54.5%
- [x] Exp1 多任务训练（摘要+主题）→ train_loss=4.42
- [x] Exp1_multi 评测 → ROUGE-L=23.48，Length Acc=54.5%

### Phase 5 - 文档整理
- [x] 更新 `CHECKPOINTS.md`（CP-01~CP-10 全部标记）
- [x] 更新 `TASKS.md`
- [x] 更新 `results/metrics/final_results.json`
- [x] 记录实验过程到 `memory/2026-03-31-experiment-log.md`

---

## Phase 6 - Qwen 跨模型实验（已完成）

### 代码准备（已完成）
- [x] `run_training.py` 支持 causal LM（Qwen）分支
- [x] `run_evaluation.py` 支持 causal LM 评测分支
- [x] `load_model.py` 加入 `prepare_causal_model()`
- [x] 修复 bug：`learning_rate` 未传入 TrainingArguments
- [x] 添加 `--skip_eval` flag（解决 MPS 内存碎片化）
- [x] 修复 Qwen eval embedding size mismatch（PeftModel 显式加载）

### 训练实验
- [x] **CP-11** exp0_qwen：300 samples × 5 epochs → ROUGE-L=34.31
- [x] **CP-12** exp1_qwen：300 samples × 5 epochs → ROUGE-L=33.41, Length Acc=68.2%
- [x] **CP-13** exp1_multi_qwen：600 samples × 5 epochs → ROUGE-L=34.29, Length Acc=74.2%

### 评测
- [x] **CP-14** Qwen 三组实验评测（n=500）
- [x] **CP-15** FLAN-T5 三组全量重评（n=1500）

## Phase 7 - 报告与可视化（已完成）
- [x] 绘制 ROUGE 对比柱状图（FLAN-T5 vs Qwen）
- [x] 绘制 Length Accuracy 分桶对比图
- [x] 整理最终对比表（final_results_v2.json 更新）
- [x] 撰写最终报告/论文（`report.md`）

---

## Notes
- 训练环境：Mac M4 24GB, MPS, FLAN-T5-base (220M), LoRA r=16
- Qwen：Qwen/Qwen3.5-0.8B, LoRA r=16, target_modules=[q_proj, v_proj], causal LM
- v1 → v2 关键修复：添加指令前缀、lr 从 5e-4 降到 5e-5、改用自然语言长度控制
- 全部 checkpoint（CP-01~CP-15）已通过
