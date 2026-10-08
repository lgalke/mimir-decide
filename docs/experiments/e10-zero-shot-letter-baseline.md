---
type: Experiment
title: 'E10: Zero-shot letter baseline (untrained Mimir)'
description: Evaluate the untrained Mimir v1.5 with the letter prompt, to show how much of the fine-tuned baseline's quality was already there.
status: planned
answers: [q14, q04]
depends_on: [e01]
tags: [baseline, zero-shot]
timestamp: 2026-10-09T00:00:00Z
---

## Hypothesis

The untrained Mimir v1.5 already answers many of these decisions with the letter prompt, because its training mix contains similar tasks ([O07](/observations/o07-mimir-policy-includes-tasksource-and-flan.md)). Fine-tuning adds a clear gain on top. The size of the gain is the unknown: it tells whether the fine-tuned baseline's accuracy of about 0.835 ([O14](/observations/o14-first-pilot-results-slot-and-baseline-tie.md)) is mostly fine-tuning or mostly prior knowledge.

## Prerequisites

The mixture from [E01](/experiments/e01-full-mixture-build.md) (`calib.parquet`, `validation.parquet`, `heldout_tasks/heldout.parquet`). No training. One GPU run of a few minutes.

## Setup

```bash
# Run from the repository root with the project environment active (uv .venv or conda).
export DATA=$HOME/mimir-decide-data/mixture-v0 RUNS=runs
PY=python

# 1. Zero-shot checkpoint: no weights are copied; they are loaded from the hub at the pinned revision.
$PY -m mimir_decide.zeroshot --config configs/train_baseline.yaml --run_dir $RUNS/letter-zeroshot

# 2. Optional temperatures (3 numbers) on the calibration half of validation.
$PY -m mimir_decide.calibrate --run_dir $RUNS/letter-zeroshot --checkpoint zero-shot --data_dir $DATA

# 3. Raw and calibrated evaluation on validation and held-out tasks (never test).
for S in validation heldout; do
  $PY -m mimir_decide.evaluate --run_dir $RUNS/letter-zeroshot --checkpoint zero-shot --data_dir $DATA --split $S --no_calibration --tag uncal
  $PY -m mimir_decide.evaluate --run_dir $RUNS/letter-zeroshot --checkpoint zero-shot --data_dir $DATA --split $S
done

# 4. Side by side with the fine-tuned runs.
$PY -m mimir_decide.compare $RUNS/letter-zeroshot $RUNS/baseline-v0 $RUNS/slot-v0 --file validation.json
$PY -m mimir_decide.compare $RUNS/letter-zeroshot --file validation_uncal.json
```

The run goes to `runs/letter-zeroshot/` in the repository ([D24](/design/d24-run-directories-in-the-repository.md)). Git tracks only its small eval files. The comparison in step 4 needs only the eval JSON files of the pilot runs, which are in the repository. Use `--limit N` on `calibrate` and `evaluate` only for smoke tests, never for reported numbers.

## Metrics to record

Accuracy, NLL, Brier and ECE, raw and calibrated, for all decisions and per kind; the pooled seen-by-Mimir and not-flagged blocks; `n_skipped_too_long`; the 3 temperatures (a value at the clamp 0.05 or 20 means the raw probabilities are very far from calibrated). Compare with the fine-tuned baseline and the slot model on the same records.

## Decision rule

Written before running. Let `gain` be the fine-tuned baseline's validation accuracy minus the zero-shot accuracy (both calibrated).

- `gain` of 10 points or more: fine-tuning adds most of the quality. State that in O14's interpretation.
- `gain` below 3 points: most of the quality was already in Mimir. State that, and treat the slot-versus-baseline comparison as a comparison of two readouts on a model that barely changed.
- Between 3 and 10 points: both contribute.

The result changes the wording of the conclusions, not the plan. [E11](/experiments/e11-stronger-baseline-trainable-lm-head.md) follows in any case.

Limit of this reference: the prompt is ours (`State / Question / Options / Answer with the letter`), not Mimir's own training format, so a low zero-shot result is a lower bound for what Mimir can do without training.

## Status and results

Status: **planned**. Results: _not run yet_.

| Date | Run id / path | Config (overrides) | Result | Observation page |
|---|---|---|---|---|
| | | | | |

When run: fill this table, write an observation page from the template, set `status`, add a line to the update log, and regenerate the indexes.
