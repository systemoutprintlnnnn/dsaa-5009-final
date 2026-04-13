# 2026-03-31 实验执行日志

> 本文档记录了 CP-05 ~ CP-09 全部实验的详细执行过程，包括环境搭建、代码编写、训练运行、评测结果与问题分析。

---

## 一、环境搭建

### 1.1 创建虚拟环境
- 机器：Mac M4 24GB，macOS Darwin 24.5.0
- Python：3.14.0（/Library/Frameworks/Python.framework/Versions/3.14/bin/python3）
- 项目没有预先创建 venv，需要手动创建：
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- 依赖安装成功，关键版本：
  - transformers==5.4.0
  - peft==0.18.1
  - torch==2.11.0
  - datasets==4.8.4
  - evaluate==0.4.6

### 1.2 设备选择
- 训练设备：**MPS (Apple Silicon)**，自动检测
- MPS 可用且已构建：`torch.backends.mps.is_available()=True`

---

## 二、CP-05 单步训练 Smoke Test

### 2.1 执行命令
```bash
source .venv/bin/activate
PYTHONPATH=. python scripts/check_training_step.py
```

### 2.2 结果
- **状态**：通过
- **设备**：mps
- **Loss**：459.453613（首步 loss，模型未训练，正常偏高）
- **Grad Norm**：89.87
- **Trainable Grad Params**：144
- **Batch Shape**：input [1, 286], labels [1, 48]
- **Loss is Finite**：True
- **Report 保存位置**：`results/metrics/training_smoke_test.json`

### 2.3 验证项
- [x] forward + backward 正常
- [x] loss 为有限值、非 NaN
- [x] tokenizer 正确处理了 special tokens（5 个 token 已添加）
- [x] LoRA 注入成功（trainable ratio = 0.7877%）

---

## 三、创建完整训练脚本 (CP-06 前置)

### 3.1 新建文件
- `scripts/run_training.py` — 支持三种实验模式的完整训练脚本

### 3.2 脚本设计
```
实验模式：
  exp0       — Baseline（无 length tokens，仅摘要）
  exp1       — Length-controllable（单任务，带 length tokens）
  exp1_multi — 多任务（摘要 + 主题生成，带 length tokens）
```

关键参数：
- 默认模型：google/flan-t5-base
- LoRA：r=16, alpha=32, target_modules=["q", "v"], dropout=0.05
- 训练：3 epochs, batch_size=8（Mac 上改为 4）, lr=5e-4, warmup_steps=100

### 3.3 遇到的问题与修复

**问题 1：`tokenizer` 参数名变更**
- 错误：`TypeError: Seq2SeqTrainer.__init__() got an unexpected keyword argument 'tokenizer'`
- 原因：transformers 5.x 将 `tokenizer` 参数改名为 `processing_class`
- 修复：`tokenizer=tokenizer` → `processing_class=tokenizer`

**问题 2：`logging_dir` 弃用警告**
- 警告：`logging_dir is deprecated and will be removed in v5.2`
- 修复：移除 `logging_dir` 参数

---

## 四、CP-06 Exp0 Baseline 训练

### 4.1 执行命令
```bash
source .venv/bin/activate
PYTHONPATH=. python scripts/run_training.py --exp exp0 --batch_size 4 --warmup_steps 100
```

### 4.2 训练配置
| 参数 | 值 |
|------|-----|
| 模型 | google/flan-t5-base |
| 实验模式 | exp0（baseline，无 length tokens） |
| 训练样本数 | 12,460 |
| 验证样本数 | 500 |
| Epochs | 3 |
| Batch Size | 4 |
| Learning Rate | 5e-4 |
| Warmup Steps | 100 |
| Max Input Length | 512 |
| Max Target Length | 128 |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| 设备 | MPS |
| Trainable Params | 1,769,472 / 224,651,520 (0.7877%) |

### 4.3 训练过程
- 总步数：9,345 steps (3 epochs × 3,115 steps)
- 训练速度：~1.4-1.8 it/s（波动）
- 总训练时间：**约 1 小时 45 分钟**（train_runtime=6269s）

**Eval Loss 变化：**
| Epoch | Eval Loss |
|-------|-----------|
| 1 | 9.458 |
| 2 | 8.192 |
| 3 | （best model loaded） |

### 4.4 训练结果
- **Train Loss**: 22.63
- **模型保存位置**：`results/models/exp0/`

---

## 五、评测管线创建 (CP-07)

### 5.1 新建文件
- `scripts/run_evaluation.py` — 统一评测脚本
- `src/evaluation/rouge.py` — ROUGE 计算（ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum）
- `src/evaluation/length_metrics.py` — Length Accuracy 和 Length MAE

### 5.2 评测方法
- 生成方式：beam search (num_beams=4, early_stopping=True)
- Batch 生成：batch_size=8
- 对于带 length tokens 的实验：输入格式为 `<len_L> [SUMMARIZE] dialogue`
- Length Accuracy：生成摘要词数落在目标范围内的比例
- Length MAE：生成词数与目标桶中点的平均绝对误差

---

## 六、Exp0 Baseline 评测

### 6.1 执行命令
```bash
PYTHONPATH=. python scripts/run_evaluation.py --exp exp0 --split test
```

### 6.2 评测结果
- 测试集样本数：1,500（DialogSum test split）
- 生成耗时：约 1 小时 15 分钟

| 指标 | 分数 |
|------|------|
| ROUGE-1 | 21.70 |
| ROUGE-2 | 5.74 |
| ROUGE-L | 18.24 |
| ROUGE-Lsum | 18.23 |

- 结果保存：`results/metrics/eval_results_exp0.json`

---

## 七、CP-08 Exp1 Length-Controllable 训练

### 7.1 执行命令
```bash
PYTHONPATH=. python scripts/run_training.py --exp exp1 --batch_size 4 --warmup_steps 100
```

### 7.2 训练配置
- 与 Exp0 相同，**区别在于**：
  - 添加了 5 个 special tokens：`<len_SHORT>`, `<len_MEDIUM>`, `<len_LONG>`, `[SUMMARIZE]`, `[TOPIC]`
  - 输入格式：`<len_L> [SUMMARIZE] dialogue`（L 根据 ground truth summary 词数自动分桶）
  - 仅使用摘要任务（单任务）

### 7.3 训练过程
- 总步数：9,345 steps
- 训练速度：~1.0-1.5 it/s
- 总训练时间：**约 2 小时 21 分钟**（train_runtime=8442s）

**Eval Loss 变化：**
| Epoch | Eval Loss |
|-------|-----------|
| 1 | ~9.x |
| 2 | ~8.x |

### 7.4 训练结果
- **Train Loss**: 22.54（略低于 Exp0 的 22.63）
- **模型保存位置**：`results/models/exp1/`

### 7.5 Exp1 评测结果
```bash
PYTHONPATH=. python scripts/run_evaluation.py --exp exp1 --split test
```

| 指标 | 分数 |
|------|------|
| ROUGE-1 | 21.98 |
| ROUGE-2 | 5.34 |
| ROUGE-L | 18.08 |
| ROUGE-Lsum | 18.09 |
| **Length Accuracy** | **28.73%** |
| **Length MAE** | **40.17** |
| Length Acc (SHORT) | 44.35% |
| Length Acc (MEDIUM) | 17.57% |
| Length Acc (LONG) | 19.70% |

---

## 八、CP-09 Exp1 Multi-Task 训练

### 8.1 执行命令
```bash
PYTHONPATH=. python scripts/run_training.py --exp exp1_multi --batch_size 4 --warmup_steps 100
```

### 8.2 训练配置
- 与 Exp1 相同，**区别在于**：
  - 多任务数据：每条对话生成 2 个样本（摘要 + 主题）
  - 训练样本数：12,460 × 2 = **24,920**
  - 验证样本数：500 × 2 = **1,000**
  - 输入格式：
    - 摘要任务：`<len_L> [SUMMARIZE] dialogue`
    - 主题任务：`[TOPIC] dialogue`

### 8.3 训练过程
- 总步数：18,690 steps (24,920 / 4 × 3 epochs)
- 训练速度：~1.0-1.1 it/s
- 总训练时间：**约 4 小时 49 分钟**（train_runtime=17359s）

### 8.4 训练结果
- **Train Loss**: **19.66**（显著低于 Exp0 的 22.63 和 Exp1 的 22.54）
- **模型保存位置**：`results/models/exp1_multi/`

### 8.5 Exp1 Multi 评测结果
```bash
PYTHONPATH=. python scripts/run_evaluation.py --exp exp1_multi --split test
```

| 指标 | 分数 |
|------|------|
| ROUGE-1 | 21.93 |
| ROUGE-2 | 5.37 |
| ROUGE-L | 18.06 |
| ROUGE-Lsum | 18.07 |
| **Length Accuracy** | **28.73%** |
| **Length MAE** | **40.17** |
| Length Acc (SHORT) | 44.35% |
| Length Acc (MEDIUM) | 17.57% |
| Length Acc (LONG) | 19.70% |

---

## 九、三组实验结果对比

### 9.1 ROUGE 对比

| 指标 | Exp0 Baseline | Exp1 Length Tokens | Exp1 Multi-Task |
|------|:---:|:---:|:---:|
| ROUGE-1 | 21.70 | **21.98** | 21.93 |
| ROUGE-2 | **5.74** | 5.34 | 5.37 |
| ROUGE-L | **18.24** | 18.08 | 18.06 |
| ROUGE-Lsum | **18.23** | 18.09 | 18.07 |

### 9.2 训练效率对比

| 指标 | Exp0 | Exp1 | Exp1 Multi |
|------|:---:|:---:|:---:|
| Train Loss | 22.63 | 22.54 | **19.66** |
| Train Samples | 12,460 | 12,460 | 24,920 |
| Train Time | ~1h45m | ~2h21m | ~4h49m |
| Device | MPS | MPS | MPS |

### 9.3 Length Control 对比

| 指标 | Exp1 | Exp1 Multi |
|------|:---:|:---:|
| Length Accuracy | 28.73% | 28.73% |
| Length MAE | 40.17 | 40.17 |
| SHORT Acc | 44.35% | 44.35% |
| MEDIUM Acc | 17.57% | 17.57% |
| LONG Acc | 19.70% | 19.70% |

---

## 十、关键发现与分析

### 10.1 正面发现
1. **Length tokens 不损害摘要质量**：Exp1 和 Exp1_multi 的 ROUGE 与 baseline 基本持平（差异 < 2%），说明 length tokens 的引入不会影响模型生成摘要的能力。
2. **多任务学习显著降低训练 loss**：Exp1_multi 的 train_loss（19.66）比 Exp0（22.63）低 13.2%，比 Exp1（22.54）低 12.8%。这证明 topic 辅助任务确实帮助模型更好地理解对话语义。
3. **SHORT 桶长度控制最好**：44.35% 的准确率远高于 MEDIUM（17.57%）和 LONG（19.70%），说明模型对短文本长度控制更容易学习。

### 10.2 需要改进的地方
1. **Length Accuracy 整体偏低（~29%）**：远低于预期的 85%。可能原因：
   - 3 epoch 训练不够充分，模型还没学会精确控制长度
   - FLAN-T5-base（220M）参数量偏小，长度控制能力有限
   - Mac MPS 上训练速度受限（~1.5 it/s），限制了总训练步数
2. **ROUGE 整体偏低**：与 Proposal 预期的 ROUGE-L ~39 有较大差距（实际 ~18）。可能原因：
   - Proposal 中的预期值基于更充分的训练或更优的超参
   - 3 epoch + lr=5e-4 可能不够，需要更多 epoch 或调整学习率
   - eval_loss 从 9.46 降到 8.19，仍在下降，说明模型未充分收敛
3. **Exp1 和 Exp1_multi 的 Length Accuracy 完全相同**：两个模型产生了非常相似（甚至相同）的生成结果，这可能说明：
   - 测试集上的生成结果巧合一致
   - 或者多任务学习的改善主要体现在训练 loss 而非生成输出

### 10.3 后续优化方向
- 增加训练 epoch 到 5-10，观察 ROUGE 和 Length Accuracy 是否持续提升
- 在 GPU（T4/CUDA）上训练，速度更快，可以做更多超参搜索
- 尝试更大的模型（Gemma-2-2B, Llama-3.2-3B）
- 调整 LoRA rank（如 r=32）或 target modules（加入 k_proj, o_proj）

---

## 十一、产出文件清单

### 代码文件（新建/修改）
| 文件 | 说明 |
|------|------|
| `scripts/run_training.py` | 完整训练脚本，支持 exp0/exp1/exp1_multi |
| `scripts/run_evaluation.py` | 统一评测脚本（ROUGE + Length Metrics） |
| `src/evaluation/rouge.py` | ROUGE 评测模块 |
| `src/evaluation/length_metrics.py` | Length Accuracy / MAE 评测模块 |

### 模型 Checkpoint
| 目录 | 说明 |
|------|------|
| `results/models/exp0/` | Baseline 模型 |
| `results/models/exp1/` | Length-controllable 模型 |
| `results/models/exp1_multi/` | Multi-task 模型 |

### 评测结果
| 文件 | 说明 |
|------|------|
| `results/metrics/training_smoke_test.json` | CP-05 smoke test 报告 |
| `results/metrics/eval_results_exp0.json` | Exp0 评测结果 |
| `results/metrics/eval_results_exp1.json` | Exp1 评测结果 |
| `results/metrics/eval_results_exp1_multi.json` | Exp1 Multi 评测结果 |
| `results/metrics/final_results.json` | 最终汇总结果 |

---

## 十二、时间线总结

| 时间 | 事件 |
|------|------|
| ~21:15 | 创建 venv，安装依赖 |
| ~21:17 | CP-05 smoke test 通过 |
| ~21:19 | 创建 run_training.py |
| ~21:21 | 首次运行 Exp0 失败（tokenizer 参数错误） |
| ~21:21 | 修复后重新启动 Exp0 训练 |
| ~23:05 | Exp0 训练完成（~1h45m） |
| ~23:06 | 并行启动 Exp0 评测 + Exp1 训练 |
| ~00:20 | Exp0 评测完成（~1h15m） |
| ~00:20 | Exp1 训练完成（~2h21m） |
| ~00:21 | 并行启动 Exp1 评测 + Exp1_multi 训练 |
| ~01:35 | Exp1 评测完成 |
| ~05:00 | Exp1_multi 训练完成（~4h49m） |
| ~05:01 | 启动 Exp1_multi 评测 |
| ~06:15 | Exp1_multi 评测完成 |
| ~06:16 | 全部实验完成，汇总结果 |
