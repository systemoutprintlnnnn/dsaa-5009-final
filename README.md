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
### Completed checkpoints
- `CP-01` Project skeleton → **PASS**
- `CP-02` Data analysis and length bucket validation → **PASS**
- `CP-03` Multi-task data pipeline validation → **PASS**
- `CP-04` Model loading verification → **PASS**

### Current focus
- `CP-05` One-batch training smoke test

---

## 7. Current Artifacts
Existing useful outputs:
- `results/metrics/data_stats.json`
- `results/metrics/length_distribution.png`
- `results/metrics/multitask_samples.json`
- `results/metrics/model_check_flan.json`

Target next artifact:
- `results/metrics/training_smoke_test.json`

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
If you want to continue the project right now, the best next command is:

```bash
PYTHONPATH=. python scripts/check_training_step.py
```

If this succeeds and produces:
- `results/metrics/training_smoke_test.json`

then `CP-05` can move toward acceptance.
