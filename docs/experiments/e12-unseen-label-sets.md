---
type: Experiment
title: 'E12: Unseen label sets (leave-tasks-out)'
description: Train the slot model and the letter baseline on a mixture without ten whole tasks, and test on those tasks, whose label sets never occur in training.
status: planned
answers: [q04, q01]
depends_on: [e01, e10]
tags: [generalisation, label-sets, slot, baseline]
timestamp: 2026-10-11T00:00:00Z
---

## Hypothesis

The slot readout was designed for open vocabulary: it reads a score at the place where each option's text sits, and the options can be anything. The letter baseline must map an option to a letter. If the design matters, the slot model transfers better to tasks whose label sets it never saw in fine-tuning. The pilot could not test this, because every validation task also occurred in training, and the held-out nb-NO task reuses the intent names of the other MASSIVE locales. Alternative: both models transfer equally well, and the simpler baseline is preferred.

## What counts as an unseen label set

A source is eligible when (after normalising the option strings) its rows repeat a fixed option set (at least 50% of rows in a set that occurs 20 times or more), at most 10% of its rows have an option set that another training source uses, at most 30% of its option strings occur in other training sources, and it has at least 100 train and 150 eval rows. This is exact-string novelty, not meaning: a task with 'good/bad' counts as new next to a task with 'positive/negative'. Per-item multiple-choice questions (different answers in every row) are excluded because they have no label set. Two of the ten sources are chosen among those that Mimir's training mix does not contain by name, so they are unseen for both our fine-tuning and Mimir.

## Prerequisites

The pilot mixture `$DATA` (with `reports/mimir_overlap.json`); the pilot checkpoints in `runs/` for the reference rows (see D24 for the copy command); the zero-shot run from [E10](/experiments/e10-zero-shot-letter-baseline.md). GPU time: two trainings of about 16 hours each (the same settings as the pilot, one seed).

## Setup

```bash
# Run from the repository root with the project environment active (uv .venv or conda).
export DATA=$HOME/mimir-decide-data/mixture-v0 DATA_UNSEEN=$HOME/mimir-decide-data/mixture-unseen-labels
PY=python

# 1. Select the sources (read-only, minutes). PRE-REGISTRATION: commit the two output files BEFORE any training.
$PY -m mimir_decide.select_heldout --data_dir $DATA --seed 0 --n_flagged 8 --n_unflagged 2 --out configs/heldout_unseen_labels.yaml
#    If fewer sources qualify, loosen --min_train/--min_eval/--max_seen/--max_overlap or lower the counts, and log the change.

# 2. Derive the mixture without those sources (no rebuild; train rows removed, their eval rows move to heldout_tasks/).
$PY -m mimir_decide.derive_mixture --src_dir $DATA --dst_dir $DATA_UNSEEN --sources_file configs/heldout_unseen_labels.yaml

# 3. Train both models, same settings as the pilot.
$PY -m mimir_decide.train --config configs/train_slot.yaml     --set data_dir=$DATA_UNSEEN run_dir=runs/slot-unseen-labels
$PY -m mimir_decide.train --config configs/train_baseline.yaml --set data_dir=$DATA_UNSEEN run_dir=runs/baseline-unseen-labels

# 4. Calibrate and evaluate the two new models (validation: label sets seen; heldout: the unseen sources and nb-NO).
for R in slot-unseen-labels baseline-unseen-labels; do
  $PY -m mimir_decide.calibrate --run_dir runs/$R --data_dir $DATA_UNSEEN
  $PY -m mimir_decide.evaluate  --run_dir runs/$R --data_dir $DATA_UNSEEN --split heldout --tag unseen
  $PY -m mimir_decide.evaluate  --run_dir runs/$R --data_dir $DATA_UNSEEN --split validation
done

# 5. Reference rows on the same held-out data: the zero-shot model (lower reference) and the pilot models,
#    which were trained on these tasks (upper reference, label sets seen).
$PY -m mimir_decide.evaluate --run_dir runs/letter-zeroshot --checkpoint zero-shot --data_dir $DATA_UNSEEN --split heldout --tag unseen
for R in slot-v0 baseline-v0; do $PY -m mimir_decide.evaluate --run_dir runs/$R --data_dir $DATA_UNSEEN --split heldout --tag unseen; done

# 6. Compare. The first run is the reference for the gain.
$PY -m mimir_decide.compare runs/letter-zeroshot runs/slot-unseen-labels runs/baseline-unseen-labels runs/slot-v0 runs/baseline-v0 \
    --file heldout_unseen.json --sources-file configs/heldout_unseen_labels.yaml
```

The held-out file contains the validation, calibration and test rows of the held-out sources, as for every held-out task ([D11](/design/d11-split-isolation-and-test-access.md)). `configs/data.yaml` and the pilot's held-out set are not changed ([D25](/design/d25-leave-tasks-out-for-unseen-label-sets.md)).

## Metrics to record

Per source and pooled over the selected sources (not nb-NO): accuracy, NLL, Brier, ECE, confidence. The gain over the zero-shot model in accuracy points. The share of the gap to the seen-label pilot models that the new models recover: (unseen model - zero-shot) / (pilot model - zero-shot). The not-flagged sources separately, with their small n. Validation accuracy of the new models (label sets seen), to check that they trained normally.

## Decision rule

Written before any model is trained. `G` is the pooled accuracy of a model minus the pooled accuracy of the zero-shot model on the selected sources.

- **Slot supported:** `G_slot - G_baseline` is at least 2.0 points **and** the slot model has the higher accuracy on at least 60% of the sources.
- **Baseline supported:** the same with the roles swapped.
- **Otherwise a tie:** the simpler baseline is preferred.
- If the slot model is supported, the baseline with a trainable output head ([E11](/experiments/e11-stronger-baseline-trainable-lm-head.md)) must be trained and tested on the same split before the conclusion is final.

The 2-point margin is about two standard errors of the difference for roughly 3,000 pooled decisions. It is a proposal; the owner may change it before training.

## Limits

One seed. The sources come from one seeded selection. Per-source n is about 150 to 400. Sources flagged by name may still be familiar to Mimir ([Q01](/questions/q01-clean-generalisation-evidence.md)); only the two not-flagged sources are unseen for both. Label novelty is exact-string, not semantic.

## Status and results

Status: **planned**. Results: _not run yet_.

| Date | Run id / path | Config (overrides) | Result | Observation page |
|---|---|---|---|---|
| | | | | |
