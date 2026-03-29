# DSAA-5009 Final Project

## Project Title
Multi-Task Learning for Length-Controllable Dialogue Summarization

## Overview
This project studies dialogue summarization with two main ideas:
- **Length control**: generate short / medium / long summaries with special tokens
- **Multi-task learning**: jointly learn summarization and topic generation

Dataset: **DialogSum**

---

## Environment Setup

### 1. Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Current Project Structure

```bash
config/
scripts/
src/
results/
TASKS.md
CHECKPOINTS.md
WORKFLOW.md
PROJECT_PLAN.md
PROPOSAL.md
```

---

## How to Run Current Scripts

### Analyze dataset
```bash
source .venv/bin/activate
python scripts/analyze_data.py
```

Outputs:
- `results/metrics/data_stats.json`
- `results/metrics/length_distribution.png`

### Check multitask data construction
```bash
source .venv/bin/activate
PYTHONPATH=. python scripts/check_multitask_data.py
```

Outputs:
- `results/metrics/multitask_samples.json`

---

## Workflow

The project follows a checkpoint-based workflow:

- `TASKS.md` → task tracking
- `CHECKPOINTS.md` → verification and acceptance
- `WORKFLOW.md` → execution protocol
- `memory/YYYY-MM-DD.md` → daily progress log

A step is considered complete only if:
1. implementation exists
2. verification passes
3. artifact is generated
4. checkpoint is updated

---

## Current Progress

Completed:
- CP-01 Project skeleton
- CP-02 Data analysis and length bucket validation
- CP-03 Multitask data pipeline validation

Next:
- CP-04 Model loading verification
