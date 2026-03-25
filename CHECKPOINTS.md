# CHECKPOINTS

> Project: DSAA-5009 Final Project - Multi-Task Learning for Length-Controllable Dialogue Summarization
> Purpose: stage-by-stage acceptance mechanism
> Last updated: 2026-03-25

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
- **Done Definition**:
  - 已创建 `config/`, `src/`, `scripts/`, `results/`
  - 已创建 `TASKS.md`
  - 已建立基础 Python package 结构
- **Verification**:
  - [x] `config/` 存在
  - [x] `src/data/` 存在
  - [x] `src/models/` 存在
  - [x] `src/training/` 存在
  - [x] `src/evaluation/` 存在
  - [x] `scripts/` 存在
  - [x] `results/metrics/` 存在
  - [x] `TASKS.md` 存在
- **Artifacts**:
  - 项目目录结构
  - `TASKS.md`
- **Result**: 基础骨架已建立，可以进入数据阶段

---

## CP-02 数据分析与长度分桶验证
- **Status**: 🟢 PASS
- **Goal**: 验证 DialogSum 数据集可读取，长度分布合理，分桶阈值可用
- **Done Definition**:
  - 能成功读取 train/validation/test
  - 输出 summary 长度统计
  - 输出 SHORT / MEDIUM / LONG 比例
  - 给出是否需要调整阈值的结论
- **Verification**:
  - [x] `scripts/analyze_data.py` 可运行
  - [x] 输出 train/validation/test 样本数
  - [x] 输出 summary 长度均值 / 中位数 / 分位数
  - [x] 输出 SHORT/MEDIUM/LONG 分桶比例
  - [x] 生成 `results/metrics/data_stats.json`
  - [x] 生成 `results/metrics/length_distribution.png`
- **Artifacts**:
  - `scripts/analyze_data.py`
  - `results/metrics/data_stats.json`
  - `results/metrics/length_distribution.png`
- **Result**: 数据读取、长度统计、分桶验证与分布图均已完成，CP-02 通过。

---

## CP-03 多任务数据管线验证
- **Status**: ⚪ PENDING
- **Goal**: 验证 summary + topic 的多任务数据构建逻辑正确
- **Done Definition**:
  - 每条原始样本能扩展为摘要任务 + 主题任务
  - 输入格式正确
  - 输出字段完整
- **Verification**:
  - [ ] 能正确生成 `[SUMMARIZE]` 样本
  - [ ] 能正确生成 `[TOPIC]` 样本
  - [ ] 长度 token 能正确注入
  - [ ] 随机抽样检查 5 条样本内容
  - [ ] 生成 `results/metrics/multitask_samples.json`
- **Artifacts**:
  - `src/data/preprocessing.py`
  - `results/metrics/multitask_samples.json`
- **Result**: Pending

---

## CP-04 模型加载验证
- **Status**: ⚪ PENDING
- **Goal**: 验证模型、tokenizer、special tokens、LoRA/QLoRA 配置可正常加载
- **Done Definition**:
  - 至少 FLAN-T5 能正常加载
  - 能添加长度 token 和任务 token
  - 能打印 trainable parameters
- **Verification**:
  - [ ] `scripts/check_model_loading.py --model flan-t5` 成功
  - [ ] tokenizer vocab 成功扩展
  - [ ] model embedding resize 成功
  - [ ] LoRA 配置成功注入
  - [ ] 生成 `results/metrics/model_check_flan.json`
- **Artifacts**:
  - `src/models/load_model.py`
  - `scripts/check_model_loading.py`
  - `results/metrics/model_check_flan.json`
- **Result**: Pending

---

## CP-05 单步训练验证
- **Status**: ⚪ PENDING
- **Goal**: 验证训练流程至少能跑通 1 batch
- **Done Definition**:
  - 能完成 forward + backward
  - loss 数值正常
  - 不出现 shape / tokenizer / label 错误
- **Verification**:
  - [ ] `scripts/check_training_step.py` 成功
  - [ ] loss 为数值且非 nan
  - [ ] checkpoint 或 log 成功输出
  - [ ] 生成 `results/metrics/training_smoke_test.json`
- **Artifacts**:
  - `src/training/trainer.py`
  - `scripts/check_training_step.py`
  - `results/metrics/training_smoke_test.json`
- **Result**: Pending

---

## CP-06 Baseline 训练完成
- **Status**: ⚪ PENDING
- **Goal**: 完成 Exp0 baseline 训练并产出可评测模型
- **Done Definition**:
  - FLAN-T5 baseline 训练完成
  - 模型 checkpoint 可加载
  - 训练日志完整
- **Verification**:
  - [ ] `scripts/run_training.py --exp exp0` 成功
  - [ ] checkpoint 文件存在
  - [ ] training log 存在
  - [ ] 生成 `results/models/exp0/`
- **Artifacts**:
  - `results/models/exp0/`
  - 训练日志
- **Result**: Pending

---

## CP-07 评测管线验证
- **Status**: ⚪ PENDING
- **Goal**: 验证 ROUGE + Length Accuracy + Topic Accuracy 能正确计算
- **Done Definition**:
  - 评测脚本能读取 prediction/reference
  - 能输出全部指标
  - 结果能保存为 json/csv
- **Verification**:
  - [ ] `scripts/run_evaluation.py` 成功
  - [ ] ROUGE 成功输出
  - [ ] Length Accuracy 成功输出
  - [ ] Topic Accuracy 成功输出
  - [ ] 生成 `results/metrics/eval_results_exp0.json`
- **Artifacts**:
  - `src/evaluation/rouge.py`
  - `src/evaluation/length_metrics.py`
  - `results/metrics/eval_results_exp0.json`
- **Result**: Pending

---

## CP-08 长度控制实验完成
- **Status**: ⚪ PENDING
- **Goal**: 完成 Exp1，验证长度控制有效
- **Done Definition**:
  - 能生成 SHORT / MEDIUM / LONG 三类摘要
  - Length Accuracy 达到预期
- **Verification**:
  - [ ] Exp1 训练完成
  - [ ] 三类摘要生成成功
  - [ ] Length Accuracy 明显高于 baseline
- **Artifacts**:
  - `results/models/exp1/`
  - `results/metrics/eval_results_exp1.json`
- **Result**: Pending

---

## CP-09 多任务学习实验完成
- **Status**: ⚪ PENDING
- **Goal**: 完成多任务学习实验并验证是否提升摘要质量
- **Done Definition**:
  - summary + topic 联合训练完成
  - topic generation 正常工作
  - ROUGE 相对单任务有提升
- **Verification**:
  - [ ] 多任务训练成功
  - [ ] Topic Accuracy 成功输出
  - [ ] ROUGE 对比单任务有提升或至少不下降
- **Artifacts**:
  - `results/models/exp1_multi/`
  - `results/metrics/eval_results_exp1_multi.json`
- **Result**: Pending

---

## CP-10 结果汇总与报告材料完成
- **Status**: ⚪ PENDING
- **Goal**: 完成图表、表格、关键结论整理
- **Done Definition**:
  - 有最终实验对比表
  - 有可直接写进报告的结论
  - Proposal / Project Plan / 实验结果一致
- **Verification**:
  - [ ] 对比表生成完成
  - [ ] 图表生成完成
  - [ ] 结论写入文档
- **Artifacts**:
  - `results/metrics/final_results.csv`
  - 图表文件
  - 更新后的文档
- **Result**: Pending
