# TASKS

> Project: DSAA-5009 Final Project - Multi-Task Learning for Length-Controllable Dialogue Summarization
> Last updated: 2026-03-24

---

## In Progress
- [ ] 开始 CP-05 单步训练验证
- [x] 创建基础配置文件
- [ ] 补充训练 smoke test

---

## Todo

### Phase 0 - 项目初始化
- [x] 创建 `README.md`
- [x] 创建 `requirements.txt`
- [ ] 创建基础配置文件：`config/models.yaml`
- [ ] 创建基础配置文件：`config/training.yaml`
- [ ] 创建 `scripts/run_training.py`
- [ ] 创建 `scripts/run_evaluation.py`
- [x] 创建 `scripts/analyze_data.py`

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
- [ ] 支持 FLAN-T5 / Gemma / Llama / Qwen
- [x] 编写特殊 token 注入逻辑
- [x] 编写 LoRA / QLoRA 配置逻辑
- [x] 编写 Trainer 封装 `src/training/trainer.py`
- [ ] 编写单任务训练入口
- [ ] 编写多任务训练入口

### Phase 3 - 评测框架
- [ ] 编写 ROUGE 评测模块 `src/evaluation/rouge.py`
- [ ] 编写 Length Accuracy 指标
- [ ] 编写 Length MAE 指标
- [ ] 编写 Topic Accuracy 指标
- [ ] 编写 BERTScore 指标
- [ ] 编写统一评测入口

### Phase 4 - 实验执行
- [ ] 跑通 Exp0 baseline
- [ ] 跑通 Exp1 单任务长度控制
- [ ] 跑通 Exp1 多任务版本
- [ ] 跑通 Gemma-2 实验
- [ ] 跑通 Llama-3.2 实验
- [ ] 跑通 Qwen3 实验
- [ ] 汇总实验结果表格

### Phase 5 - 分析与文档
- [ ] 绘制长度分布图
- [ ] 绘制实验结果对比表
- [ ] 写实验分析结论
- [ ] 更新 `PROJECT_PLAN.md`
- [ ] 更新 `PROPOSAL.md`
- [ ] 整理最终报告材料

---

## Done
- [x] 确定项目方向：长度可控对话摘要
- [x] 确定创新点二：多任务学习（summary + topic）
- [x] 完成项目 Proposal (`PROPOSAL.md`)
- [x] 完成项目方案文档 (`PROJECT_PLAN.md`)
- [x] 初始化 agent identity / user context

---

## Notes
- 当前计算资源：T4 16G
- 优先顺序：先跑通 FLAN-T5 baseline，再扩展到 Gemma / Llama / Qwen
- 先保证 pipeline 可运行，再考虑优化实验
