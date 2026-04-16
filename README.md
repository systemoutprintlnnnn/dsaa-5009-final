# DSAA-5009 Final Project

## Project Title
Multi-Task Learning for Length-Controllable Dialogue Summarization

---

## 1. What This Project Is
This project is the DSAA-5009 final project for controllable dialogue summarization.

Current core idea:
- **Length control**: generate summaries with explicit target length buckets
- **Multi-task learning**: jointly train on
  - dialogue summarization
  - topic generation

Current dataset:
- `knkarthick/dialogsum`

Current preferred baseline model:
- `google/flan-t5-base`

---

## 2. Important Project Decisions
These are the current fixed decisions for the project.

### Task design
- Main task: dialogue summarization
- Auxiliary task: topic generation

### Length bucket rules
- `SHORT <= 15`
- `MEDIUM 16-35`
- `LONG >= 36`

### Input format
- Summarization task:
  - `<len_*> [SUMMARIZE] {dialogue}`
- Topic task:
  - `[TOPIC] {dialogue}`

### Special tokens
- `<len_SHORT>`
- `<len_MEDIUM>`
- `<len_LONG>`
- `[SUMMARIZE]`
- `[TOPIC]`

---

## 3. Target Runtime Environment
### Primary training machine
- **Apple Silicon Mac M4 24GB**

### Runtime policy
- This project is now maintained as a **pure Python repository**.
- Do **not** assume CUDA / NVIDIA.
- Prefer:
  1. Apple Silicon friendly execution
  2. `mps` when available
  3. `cpu` fallback when necessary

### Practical implication
When writing or updating code for this repo:
- do not hardcode CUDA
- do not assume Linux-only GPU libraries
- keep training configs lightweight enough for local Apple device testing

---

## 4. Repository Structure
```bash
config/
scripts/
src/
results/
memory/
TASKS.md
CHECKPOINTS.md
WORKFLOW.md
PROJECT_PLAN.md
PROPOSAL.md
README.md
```

### Important folders
- `scripts/` → runnable entry scripts
- `src/data/` → preprocessing logic
- `src/models/` → model loading / token injection / LoRA setup
- `src/training/` → training utilities
- `results/metrics/` → generated metrics / smoke test outputs
- `memory/` → progress logs and decisions

---

## 5. Quick Start (Recommended)
This is the shortest path for a human or another AI agent to understand and run the repo.

### Step 1: clone repo
```bash
git clone https://github.com/systemoutprintlnnnn/dsaa-5009-final.git
cd dsaa-5009-final
```

### Step 2: create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: run the existing checks
#### A. Data analysis
```bash
python scripts/analyze_data.py
```
Expected outputs:
- `results/metrics/data_stats.json`
- `results/metrics/length_distribution.png`

#### B. Multi-task data check
```bash
PYTHONPATH=. python scripts/check_multitask_data.py
```
Expected output:
- `results/metrics/multitask_samples.json`

#### C. Model loading check
```bash
PYTHONPATH=. python scripts/check_model_loading.py --model flan-t5
```
Expected output:
- `results/metrics/model_check_flan.json`

#### D. Training smoke test (CP-05 target)
```bash
PYTHONPATH=. python scripts/check_training_step.py
```
Expected output:
- `results/metrics/training_smoke_test.json`

---

## 6. Current Verified Progress
### Completed checkpoints (CP-01 ~ CP-17)
- `CP-01` ~ `CP-10` FLAN-T5 baseline + length control + multi-task → **PASS**
- `CP-11` ~ `CP-13` Qwen3.5-0.8B training (baseline, length control, multi-task) → **PASS**
- `CP-14` ~ `CP-15` Full evaluation (Qwen 500 samples, FLAN-T5 1500 samples) → **PASS**
- `CP-16` Hyperparameter systematic search (12 experiments, 6 ablation dimensions) → **PASS**
- `CP-17` Qwen full-dataset training on A100 with optimal config → **PASS**

### Best results
| Model | Experiment | ROUGE-1 | ROUGE-L | Length Acc | Samples | Device |
|-------|-----------|:-------:|:-------:|:----------:|:-------:|--------|
| FLAN-T5-base | Exp1_multi | 30.94 | 26.05 | 47.8% | 12,460 | Mac M4 |
| Qwen3.5-0.8B | Exp0 (300) | 42.21 | **34.31** | — | 300 | Mac M4 |
| Qwen3.5-0.8B | Exp1_multi (600) | 42.24 | 34.29 | **74.2%** | 600 | Mac M4 |
| Qwen3.5-0.8B | Exp0 Full (v3) | 36.01 | 30.21 | — | 12,460 | A100 |
| Qwen3.5-0.8B | Exp1 Multi Full (v3) | 34.66 | 29.33 | 61.2% | 24,920 | A100 |

### Optimal Qwen LoRA config (from hparam search)
| Parameter | Value |
|-----------|-------|
| LoRA modules | q_proj, v_proj (2) |
| Rank / Alpha | 16 / 32 |
| Learning rate | 5e-5 |
| Epochs | 5 |
| Effective batch size | 2 |
| Prompt format | Simple text |

---

## 7. Current Artifacts
Existing useful outputs:
- `results/metrics/` — FLAN-T5 evaluation results
- `results/local_validation/` — Qwen evaluation results (v2)
- `results/exp*_qwen_full/` — Qwen evaluation results (v2, 300 samples)
- `results/exp*_qwen_v3/` — Qwen full-dataset results (v3, 12,460 samples, A100)
- `results/hparam_search/` — 12 hyperparameter ablation experiments
- `docs/hparam_exploration.md` — Complete hyperparameter search documentation
- `report.md` / `report/main.tex` — Final report

---

## 8. How the Project Is Managed
This repo uses a checkpoint-based workflow.

### Key files
- `TASKS.md`
  - task tracking
- `CHECKPOINTS.md`
  - formal checkpoint acceptance status
- `WORKFLOW.md`
  - project execution rules
- `memory/YYYY-MM-DD.md`
  - daily progress notes and decisions

### Completion rule
A step is complete only if all 4 are true:
1. code exists
2. code runs
3. output matches expectation
4. artifact is saved

---

## 9. Guidance for AI Agents
If you are an AI agent entering this repository, do this first:

1. Read `README.md`
2. Read `TASKS.md`
3. Read `CHECKPOINTS.md`
4. Read `WORKFLOW.md`
5. Read the latest file in `memory/`

### Then follow this order
1. understand current checkpoint
2. make the smallest useful change
3. run the smallest possible verification
4. save artifact
5. update docs if the project state changed

### Do not assume
- do not assume CUDA exists
- do not assume Colab is the execution path
- do not assume new data collection is allowed
- do not change evaluation set casually

### Preferred engineering style
- pure Python scripts first
- small verification scripts
- explicit artifact outputs
- Mac-friendly runtime choices

---

## 10. Current Engineering Constraints
- Main machine: **Mac M4 24GB**
- Prefer Apple-friendly execution
- Keep training settings conservative
- Prioritize a working, reproducible project over an over-ambitious model choice

---

## 11. Recommended Next Action
All planned experiments and documentation are complete. Potential future work:
- Controlled full-dataset training with 6 LoRA modules (to deconfound data size vs module count)
- Bucket-stratified sampling for better LONG accuracy
- Extension to other decoder-only models (Llama, Gemma)
