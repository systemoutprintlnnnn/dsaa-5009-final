# DSAA-5009 Final Project: 长度可控对话摘要

> **项目类型**: 对话文本摘要 + 长度控制
> **创建日期**: 2026-03-23
> **状态**: 规划中

---

## 1. 项目背景

### 1.1 任务定义
**对话文本摘要 (Dialogue Summarization)**
- **输入**: 多轮对话文本
- **输出**: 简洁的摘要

### 1.2 应用场景
- 客服对话快速浏览
- 会议记录压缩
- 社交聊天摘要
- 邮件主题生成

### 1.3 参考项目
- [dialogue-text-summarization](https://github.com/dtruong46me/dialogue-text-summarization)
- 方法: FLAN-T5 / BART + LoRA/QLoRA
- 数据集: DialogSum

---

## 2. 核心创新点

### 2.1 长度可控摘要生成 (Length-Controllable Summarization)

通过特殊 token 实现细粒度长度控制：

```
<len_SHORT> + dialogue → 5-15 words 摘要 (1句话)
<len_MEDIUM> + dialogue → 16-35 words 摘要 (2-3句)
<len_LONG> + dialogue → 36+ words 摘要 (详细版)
```

### 2.2 创新优势

| 特点 | 说明 |
|------|------|
| **显式控制** | 通过 token 精确控制输出长度，而非简单的 prompt 拼接 |
| **零成本** | 无需额外标注，基于现有 summary 长度自动分桶 |
| **实用性** | 用户可按需选择详细程度，满足不同场景需求 |
| **通用性** | 可迁移到多种基座模型 (Encoder-Decoder & Decoder-Only) |

### 2.3 与原项目对比

| 对比项 | 原项目 | 本项目 |
|--------|--------|--------|
| 长度控制 | prompt 拼接 "should be X words long" | 特殊 token 显式控制 |
| 模型 | FLAN-T5, BART | FLAN-T5, Gemma-2, Llama-3.2, Qwen3 |
| 评测 | ROUGE | ROUGE + Length Accuracy + BERTScore |
| 创新点 | Contrastive Loss | 长度可控 + 跨模型验证 |

---

## 3. 数据集

### 3.1 DialogSum 数据集

**来源**: HuggingFace `knkarthick/dialogsum`

| 指标 | 数值 |
|------|------|
| 总样本 | 13,460 + 100 holdout |
| 训练集 | 12,460 |
| 验证集 | 500 |
| 测试集 | 500 |
| 语言 | 英文 |
| 字段 | dialogue, summary, topic |

### 3.2 数据处理流程

```python
# 按 summary 长度自动分桶
def get_length_token(summary):
    word_count = len(summary.split())
    if word_count <= 15:
        return "<len_SHORT>"
    elif word_count <= 35:
        return "<len_MEDIUM>"
    else:
        return "<len_LONG>"

# 构建训练样本
def preprocess(sample):
    length_token = get_length_token(sample["summary"])
    
    input_text = f"{length_token} Summarize the following dialogue:\n###\n{sample['dialogue']}\n###\nSummary:"
    target_text = sample["summary"]
    
    return {"input": input_text, "target": target_text}
```

### 3.3 预期长度分布 (待验证)

| 长度类别 | 词数范围 | 预估比例 |
|----------|----------|----------|
| SHORT | 5-15 | ~30% |
| MEDIUM | 16-35 | ~50% |
| LONG | 36+ | ~20% |

---

## 4. 模型方案

### 4.1 模型选择

| 实验 | 模型 | 参数量 | 微调方法 | 显存 | 说明 |
|------|------|--------|----------|------|------|
| **Exp0** | google/flan-t5-base | 220M | LoRA | ~3G | Baseline (无 Length Token) |
| **Exp1** | google/flan-t5-base | 220M | LoRA + Length Token | ~3G | 验证方法有效性 |
| **Exp2** | google/gemma-2-2b-it | 2B | LoRA + Length Token | ~6G | 小参数对比 |
| **Exp3** | meta-llama/Llama-3.2-3B-Instruct | 3B | LoRA + Length Token | ~10G | Llama 家族 |
| **Exp4** | Qwen/Qwen3-4B-Instruct-2507 | 4B | QLoRA + Length Token | ~12G | 国产 SOTA |

### 4.2 模型选择理由

| 模型 | 选择理由 |
|------|----------|
| FLAN-T5 | 复现原项目，Encoder-Decoder 架构代表 |
| Gemma-2-2B | Google 最新，基于 Gemini 技术，轻量高效 |
| Llama-3.2-3B | Meta 最新小模型，5.9M 下载量，质量可靠 |
| Qwen3-4B | 通义千问最新版，5.6M 下载量，中文友好 |

### 4.3 显存估算 (T4 16G)

| 模型 | 量化 | 训练显存 | 可行性 |
|------|------|----------|--------|
| FLAN-T5 (220M) | 无 | ~2-3G | ✅ 非常轻松 |
| Gemma-2-2B | 无 | ~5-6G | ✅ 轻松 |
| Llama-3.2-3B | 无 | ~8-10G | ✅ OK |
| Qwen3-4B | 4-bit | ~10-12G | ✅ 可行 |

---

## 5. 评测指标

### 5.1 主指标 (复用原项目)

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| ROUGE-1 | 单词重叠率 | unigram matching |
| ROUGE-2 | 双词重叠率 | bigram matching |
| ROUGE-L | 最长公共子序列 | LCS matching |
| ROUGE-Lsum | 摘要级别 | sentence-level LCS |

### 5.2 新增指标 (长度控制效果)

| 指标 | 定义 | 公式 |
|------|------|------|
| **Length Accuracy** | 输出长度落在目标范围内的比例 | `count(len ∈ target_range) / total` |
| **Length MAE** | 预测长度与目标长度的平均绝对误差 | `mean(|pred_len - target_len|)` |
| **Cross-Length Consistency** | 不同长度摘要的语义一致性 | BERTScore(SHORT, MEDIUM, LONG) |

### 5.3 评测代码框架

```python
import evaluate

# ROUGE 评测
rouge = evaluate.load("rouge")
rouge_results = rouge.compute(
    predictions=generated_summaries,
    references=reference_summaries,
    use_stemmer=True
)

# Length Accuracy
def length_accuracy(predictions, targets, range_map):
    correct = 0
    for pred, target_token in zip(predictions, targets):
        pred_len = len(pred.split())
        target_range = range_map[target_token]  # e.g., (5, 15)
        if target_range[0] <= pred_len <= target_range[1]:
            correct += 1
    return correct / len(predictions)

# BERTScore (语义一致性)
bertscore = evaluate.load("bertscore")
consistency = bertscore.compute(
    predictions=short_summaries,
    references=long_summaries,
    lang="en"
)
```

---

## 6. 实验设计

### 6.1 主实验: 长度控制有效性

**目标**: 证明 Length Token 能有效控制输出长度且不损失质量

| 对比 | Exp0 vs Exp1-4 |
|------|----------------|
| 自变量 | 是否使用 Length Token |
| 因变量 | ROUGE + Length Accuracy |

**假设**:
- H1: Exp1-4 的 Length Accuracy 显著高于 Exp0
- H2: Exp1-4 的 ROUGE 不低于 Exp0

### 6.2 跨模型对比

**目标**: 分析哪种模型对长度控制更敏感

| 对比 | Exp1 vs Exp2 vs Exp3 vs Exp4 |
|------|------------------------------|
| 自变量 | 基座模型类型 |
| 因变量 | Length Accuracy + ROUGE |

**分析维度**:
- Encoder-Decoder (FLAN-T5) vs Decoder-Only (Gemma/Llama/Qwen)
- 参数量影响 (220M vs 2B vs 3B vs 4B)

### 6.3 Ablation Study

**目标**: 观察单一长度训练对其他长度生成的影响

| 实验 | 训练数据 | 测试数据 |
|------|----------|----------|
| Ablation-1 | 仅 SHORT | SHORT/MEDIUM/LONG |
| Ablation-2 | 仅 MEDIUM | SHORT/MEDIUM/LONG |
| Ablation-3 | 仅 LONG | SHORT/MEDIUM/LONG |

**预期发现**:
- 单一长度训练可能导致其他长度生成失败
- 证明多长度混合训练的必要性

### 6.4 定性分析

**人工评估维度**:

| 维度 | 问题 | 评分 (1-5) |
|------|------|------------|
| **核心信息保留** | 短摘要是否保留了对话的核心信息? | |
| **信息完整性** | 长摘要是否包含了所有重要细节? | |
| **冗余度** | 长摘要是否存在不必要的重复? | |
| **信息增量** | SHORT→MEDIUM→LONG 的信息增量是否合理? | |
| **流畅性** | 摘要是否自然流畅? | |

---

## 7. 技术实现

### 7.1 项目结构

```
dsaa-5009-final/
├── PROJECT_PLAN.md          # 本文件
├── README.md                # 项目说明
├── requirements.txt         # 依赖
├── config/
│   ├── models.yaml          # 模型配置
│   └── training.yaml        # 训练配置
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py       # 数据集处理
│   │   └── preprocessing.py # 预处理
│   ├── models/
│   │   ├── __init__.py
│   │   └── load_model.py    # 模型加载
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py       # 训练逻辑
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── rouge.py         # ROUGE 评测
│   │   └── length_metrics.py # 长度指标
│   └── utils.py
├── scripts/
│   ├── run_training.py      # 训练脚本
│   ├── run_evaluation.py    # 评测脚本
│   └── analyze_data.py      # 数据分析
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── results/
│   ├── metrics/             # 评测结果
│   └── models/              # 模型 checkpoint
└── memory/                  # 工作日志
    └── 2026-03-23.md
```

### 7.2 添加特殊 Token

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Encoder-Decoder (FLAN-T5)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# 添加长度控制 token
length_tokens = ["<len_SHORT>", "<len_MEDIUM>", "<len_LONG>"]
num_added = tokenizer.add_tokens(length_tokens)
model.resize_token_embeddings(len(tokenizer))

print(f"Added {num_added} tokens")
```

### 7.3 LoRA 配置

```python
from peft import LoraConfig, TaskType

# Encoder-Decoder (FLAN-T5)
lora_config_ed = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

# Decoder-Only (Gemma/Llama/Qwen)
lora_config_dec = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
```

### 7.4 训练参数

```python
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    predict_with_generate=True,
    generation_max_length=128,
    fp16=True,  # T4 支持
)
```

---

## 8. 时间线 (4 周)

| 周 | 任务 | 产出 | 状态 |
|----|------|------|------|
| **W1** | 环境搭建 + 数据处理 + Exp0 Baseline | 可运行的 pipeline | ⬜ |
| **W2** | 实现 Length Token + Exp1 | 初步结果 | ⬜ |
| **W3** | Exp2-4 训练 + 评测体系 | 完整实验数据 | ⬜ |
| **W4** | Ablation + 分析 + 报告 | 最终报告 | ⬜ |

### W1 详细任务
- [ ] 搭建 Google Colab / Kaggle 环境
- [ ] 下载数据集并分析分布
- [ ] 实现 Exp0 (复现原项目 baseline)
- [ ] 跑通评测流程

### W2 详细任务
- [ ] 实现特殊 token 添加
- [ ] 修改数据处理流程
- [ ] 训练 Exp1
- [ ] 对比 Exp0 vs Exp1 结果

### W3 详细任务
- [ ] 训练 Exp2 (Gemma-2-2B)
- [ ] 训练 Exp3 (Llama-3.2-3B)
- [ ] 训练 Exp4 (Qwen3-4B)
- [ ] 完善评测代码

### W4 详细任务
- [ ] Ablation Study
- [ ] 定性分析 (人工评估)
- [ ] 整理实验结果
- [ ] 撰写报告/论文

---

## 9. 预期成果

### 9.1 论文贡献

1. **显式长度控制机制**
   - 通过特殊 token 实现细粒度长度控制
   - 优于简单的 prompt 拼接方法

2. **零成本数据构建**
   - 无需额外人工标注
   - 基于现有 summary 长度自动分桶

3. **跨模型验证**
   - 在 4 种 SOTA 模型上验证方法通用性
   - 分析不同架构对长度控制的敏感度

4. **完整评测体系**
   - ROUGE (摘要质量)
   - Length Accuracy (长度控制)
   - BERTScore (语义一致性)

### 9.2 可交付物

- [ ] 训练代码 (基于 HuggingFace Transformers)
- [ ] 评测代码 (ROUGE + Length Accuracy)
- [ ] 4 个微调后的模型 (上传 HuggingFace Hub)
- [ ] 实验结果表格和图表
- [ ] 最终报告/论文

---

## 10. 计算资源

| 资源 | 规格 |
|------|------|
| **GPU** | NVIDIA T4 16GB |
| **平台** | Google Colab Pro / Kaggle / 本地 |
| **预计总训练时长** | ~20-30 小时 |
| **存储需求** | ~50GB (模型 + 数据 + 日志) |

### 训练时间估算

| 模型 | 单 epoch 时间 | 3 epochs 总时间 |
|------|--------------|-----------------|
| FLAN-T5 | ~30 min | ~1.5 h |
| Gemma-2-2B | ~1 h | ~3 h |
| Llama-3.2-3B | ~2 h | ~6 h |
| Qwen3-4B | ~3 h | ~9 h |

---

## 11. 风险与备选方案

### 11.1 潜在风险

| 风险 | 可能性 | 影响 | 应对方案 |
|------|--------|------|----------|
| Qwen3-4B 显存不足 | 中 | 高 | 使用 QLoRA 4-bit 量化 |
| Length Token 不生效 | 低 | 高 | 增加训练 epoch 或调整学习率 |
| 数据分布不均匀 | 中 | 中 | 调整分桶阈值或过采样 |
| ROUGE 下降 | 低 | 中 | 调整 LoRA rank 或增加训练数据 |

### 11.2 备选模型

如果上述模型不可用，可替换为：

| 原模型 | 备选模型 | 参数量 |
|--------|----------|--------|
| Gemma-2-2B | Phi-3-mini | 3.8B |
| Llama-3.2-3B | MiniCPM-2B | 2B |
| Qwen3-4B | Qwen2.5-7B (QLoRA) | 7B |

---

## 12. 参考资料

### 12.1 论文
- [DialogSum: A Real-Life Scenario Dialogue Summarization Dataset](https://arxiv.org/pdf/2105.06762.pdf)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

### 12.2 代码
- [dialogue-text-summarization](https://github.com/dtruong46me/dialogue-text-summarization) - 参考项目
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)

### 12.3 数据集
- [DialogSum on HuggingFace](https://huggingface.co/datasets/knkarthick/dialogsum)

---

## 13. 更新日志

| 日期 | 更新内容 |
|------|----------|
| 2026-03-23 | 创建项目方案 |

---

*本文档将随项目进展持续更新*
