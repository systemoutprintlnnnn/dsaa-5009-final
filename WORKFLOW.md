# WORKFLOW

> Execution and verification workflow for DSAA-5009 Final Project
> Last updated: 2026-03-25

---

## Core Principle

Every step must answer 3 questions:

1. **What was done?**
2. **How was it verified?**
3. **Where is the evidence?**

If a step has no verification and no artifact, it is **not considered complete**.

---

## Required Structure for Every Step

Each task should include 4 elements:

### 1. Task
What needs to be implemented.

### 2. Done Definition
What counts as completed.

### 3. Verification
How to check whether the step is correct.

### 4. Artifact
What files, outputs, logs, or figures should be produced.

---

## Step Completion Protocol

Whenever a step is finished, follow this protocol:

### Step A - Implement
Write code / config / script.

### Step B - Verify
Run the smallest possible verification script.

### Step C - Save Artifact
Save outputs to `results/metrics/`, `results/models/`, or relevant folders.

### Step D - Update Checkpoint
Update `CHECKPOINTS.md` with one of:
- 🟢 PASS
- 🟡 PARTIAL
- 🔴 FAIL

### Step E - Update Memory / Tasks
- Move completed item in `TASKS.md`
- Record conclusions / risks in `memory/YYYY-MM-DD.md`

---

## Acceptance Rules

### A step is PASS only if:
- Code exists
- Code runs
- Output matches expectation
- Artifact is saved

### A step is PARTIAL if:
- Main logic works
- But some risk / warning remains

### A step is FAIL if:
- Code does not run
- Output is clearly wrong
- Artifact missing
- Blocking issue remains unresolved

---

## Recommended Verification Scripts

To avoid hidden problems, create small dedicated checks:

- `scripts/analyze_data.py` → verify dataset and bucket distribution
- `scripts/check_model_loading.py` → verify tokenizer/model/token injection
- `scripts/check_training_step.py` → verify one-batch training
- `scripts/run_evaluation.py` → verify metric calculation

These scripts should be small, fast, and focused.

---

## Artifact Policy

Every important step should leave evidence.

### Examples

#### Data Step
- `results/metrics/data_stats.json`
- `results/metrics/length_distribution.png`

#### Model Step
- `results/metrics/model_check_flan.json`

#### Training Step
- `results/metrics/training_smoke_test.json`
- checkpoint folder under `results/models/`

#### Evaluation Step
- `results/metrics/eval_results_exp0.json`

---

## Practical Rule

Do **not** move to the next major phase unless the current checkpoint is at least:

- 🟢 PASS, or
- 🟡 PARTIAL with explicitly accepted risk

---

## Example

### Example: Data Analysis Step

**Task**
- Analyze DialogSum summary length distribution

**Done Definition**
- Dataset loads successfully
- Length stats are computed
- Bucket ratios are reported

**Verification**
- `python scripts/analyze_data.py`
- Script exits successfully
- JSON and plot are generated

**Artifact**
- `results/metrics/data_stats.json`
- `results/metrics/length_distribution.png`

**Checkpoint Result**
- Update `CHECKPOINTS.md` → CP-02

---

## Goal of This Workflow

This workflow prevents:
- fake completion
- silent bugs
- missing evidence
- carrying broken outputs into later phases

The project should progress through **verified checkpoints**, not just unfinished TODOs.
