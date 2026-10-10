---
type: Experiment
title: 'E12: Unseen label sets (evaluation on additional tasks)'
description: Evaluate all existing models, without retraining, on extra tasks whose label sets never occur in the training mixture.
status: planned
answers: [q04, q01]
depends_on: [e11]
tags: [generalisation, label-sets, slot, baseline]
timestamp: 2026-10-11T00:00:00Z
---

## Hypothesis

The slot readout was designed for open vocabulary: it reads a score at the place where each option's text sits, and the options can be anything. The letter baseline must map an option to a letter. If the design matters, the slot model transfers better to tasks whose label sets it never saw in fine-tuning. The pilot could not test this: every validation task also occurs in training, and `massive/nb-NO` reuses the intent names of the other MASSIVE locales. Alternative: all models transfer equally well, and the simpler baseline is preferred.

## Design (changed on 2026-10-11, see D26)

**No retraining.** Extra tasks that are not in the training mixture are collected as an evaluation-only set. Every model is tested on the same tasks: the zero-shot model, `slot-v0`, `baseline-v0` and the baseline with a trainable output head from [E11](/experiments/e11-stronger-baseline-trainable-lm-head.md). This keeps all runs comparable with the pilot and costs minutes of evaluation. The earlier leave-tasks-out design (retraining two models on a reduced mixture, with `select_heldout` and `derive_mixture`) is kept as a fallback only: it needs about 33 GPU hours and makes the runs incomparable with the pilot.

**Where the extra tasks come from** (the owner decides; see the open choice in D26):

- **A. Sources excluded from training only by the strict licence review.** Many bekko subsets have a permissive licence but the review status 'qualified', and some tasksource sources are on the allowlist but were not in the mixture. Using them for evaluation only would extend D02 and D13 from 'train and evaluate' to 'train'. They are mostly flagged as seen by Mimir, but they are unseen by our fine-tuning.
- **B. Other permissive datasets** that neither tasksource nor bekko carries. Needs a converter for each.
- **C. New tasks** that Mimir cannot have seen: written or labelled for this purpose, or released after Mimir's cutoff. Small, but the only clean test ([Q01](/questions/q01-clean-generalisation-evidence.md)).

**Which candidates qualify.** The same measured rule as before, applied against the training mixture: a fixed label set (at least 50% of rows in an option set that occurs 20 times or more), at most 10% of rows with an option set that a training source uses, at most 30% of option strings that occur in training options, and enough rows (at least 150). In addition, every candidate row is checked against the training data with the leakage guards (exact state hash, group id, n-gram fingerprints). Option novelty is exact-string, not semantic.

**Tooling status.** The scoring in `mimir_decide.select_heldout` can be reused for this. The script that builds the extra evaluation file (reads the candidates, applies the novelty rule and the leakage guards against `train.parquet`, writes one parquet file) is not written yet. It waits for the owner's decision on the sources.

## Setup

1. Build the extra evaluation file (tool to be written; output `$DATA_EXTRA/extra_eval.parquet` plus a list of its sources in `configs/extra_eval_sources.yaml`). Commit the source list before any model is evaluated on it.
2. Evaluate every model on it (`--split` for the extra file will be added with the tool), with the tag `extra`.
3. `python -m mimir_decide.compare runs/letter-zeroshot runs/slot-v0 runs/baseline-v0 runs/baseline-lmhead --file extra.json --sources-file configs/extra_eval_sources.yaml` (the first run is the reference for the gain).

## Metrics to record

Per source and pooled: accuracy, NLL, Brier, ECE, confidence. The gain over the zero-shot model in accuracy points. Seen-by-Mimir and not-flagged sources separately. The pilot validation accuracy as the 'label sets seen' reference.

## Decision rule

Written before any model is evaluated on the extra tasks. `G` is the pooled accuracy of a model minus the pooled accuracy of the zero-shot model.

- **Slot supported:** `G_slot - G_best_baseline` is at least 2.0 points **and** the slot model has the higher accuracy on at least 60% of the sources. `G_best_baseline` is the larger of the frozen-head and the trainable-head baseline.
- **Baseline supported:** the best baseline beats the slot model by the same margins.
- **Otherwise a tie:** the simpler baseline is preferred.

The 2-point margin is about two standard errors for roughly 3,000 pooled decisions. It is a proposal; the owner may change it before the evaluation.

## Limits

One seed per model. Per-source n is limited. Sources flagged by name may be familiar to Mimir, and only option C is unseen for both Mimir and our fine-tuning. Label novelty is exact-string, not semantic.

## Status and results

Status: **planned**. Results: _not run yet_.

| Date | Run id / path | Config (overrides) | Result | Observation page |
|---|---|---|---|---|
| | | | | |
