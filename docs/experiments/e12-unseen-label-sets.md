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

**Where the extra tasks come from.** The owner chose option A on 2026-10-11 ([D26](/design/d26-evaluate-unseen-label-sets-on-additional-tasks.md)); B and C remain possible additions:

- **A. Sources excluded from training only by the strict licence review (chosen).** Many bekko subsets have a permissive licence but the review status 'qualified', and some tasksource sources are on the allowlist but were not in the mixture. Using them for evaluation only would extend D02 and D13 from 'train and evaluate' to 'train'. They are mostly flagged as seen by Mimir, but they are unseen by our fine-tuning.
- **B. Other permissive datasets** that neither tasksource nor bekko carries. Needs a converter for each.
- **C. New tasks** that Mimir cannot have seen: written or labelled for this purpose, or released after Mimir's cutoff. Small, but the only clean test ([Q01](/questions/q01-clean-generalisation-evidence.md)).

**Which candidates qualify.** The same measured rule as before, applied against the training mixture: a fixed label set (at least 50% of rows in an option set that occurs 20 times or more), at most 10% of rows with an option set that a training source uses, at most 30% of option strings that occur in training options, and enough rows (at least 150). In addition, every candidate row is checked against the training data with the leakage guards (exact state hash, group id, n-gram fingerprints). Option novelty is exact-string, not semantic.

**Two kinds of unseen.** The novelty rule compares option strings. A bekko subset such as `bekko/snli` can therefore pass although the same task (SNLI) is in training under `tasksource/snli` with differently worded labels. That tests the shift in label wording on a familiar task, which is useful but not the same as a new task. The tool reports a name-based flag `familiar_task_in_training` and writes two lists: `held_out_sources` (all chosen: new label sets) and `task_novel_sources` (the subset with no same-name task in training). Both are evaluated and reported.

**Tooling.** `python -m mimir_decide.build_extra_eval` builds the evaluation file (licence limits of option A, exclusion of training sources, exact and n-gram leakage guards against `train.parquet`, the novelty rule). A dry run against a small trial mixture found 106 candidate sources with rows (all from bekko; every allowlisted tasksource source is already in the mixture), and dropped unknown (9,808 rows), non-commercial (5,804) and not-allowlisted (5,966) licences. Those numbers do not predict the yield against the real training mixture, which has many more option sets.

## Setup

```bash
# Run from the repository root with the project environment active (uv .venv or conda).
export DATA=$HOME/mimir-decide-data/mixture-v0 DATA_EXTRA=$HOME/mimir-decide-data/mixture-extra-eval
PY=python

# 1. Build the extra evaluation set (needs network for the bekko and tasksource eval splits; minutes).
#    PRE-REGISTRATION: commit configs/extra_eval_sources.yaml and the report summary BEFORE evaluating any model.
$PY -m mimir_decide.build_extra_eval --pilot_dir $DATA --out_dir $DATA_EXTRA --sources_out configs/extra_eval_sources.yaml
$PY -m mimir_decide.audit_mimir_overlap --data_dir $DATA_EXTRA          # flags sources that Mimir's mix contains by name

# 2. Evaluate every model on it (no training; the zero-shot model needs --checkpoint zero-shot).
$PY -m mimir_decide.evaluate --run_dir runs/letter-zeroshot --checkpoint zero-shot --data_dir $DATA_EXTRA --split extra
for R in slot-v0 baseline-v0 baseline-lmhead letter-probe slot-probe; do      # whichever of these exist (E11, E13)
  $PY -m mimir_decide.evaluate --run_dir runs/$R --data_dir $DATA_EXTRA --split extra
done

# 3. Compare. The first run is the reference for the gain. Run it for both source lists.
$PY -m mimir_decide.compare runs/letter-zeroshot runs/slot-v0 runs/baseline-v0 runs/baseline-lmhead --file extra.json \
    --sources-file configs/extra_eval_sources.yaml
$PY -m mimir_decide.compare runs/letter-zeroshot runs/slot-v0 runs/baseline-v0 runs/baseline-lmhead --file extra.json \
    --sources-file configs/extra_eval_sources.yaml --sources-key task_novel_sources
```

The extra rows are evaluation data only: no calibration, no model selection, no training ([D26](/design/d26-evaluate-unseen-label-sets-on-additional-tasks.md)). The calibrated numbers use each run's existing temperatures. Commit the small `runs/*/eval/extra.json` files.

## Metrics to record

Per source and pooled, for both source lists: accuracy, NLL, Brier, ECE, confidence. The gain over the zero-shot model in accuracy points. Per kind (noul, choice, score) for the whole file, because binary tasks are the majority of the sources. Seen-by-Mimir and not-flagged sources separately (expected: all flagged). The pilot validation accuracy as the 'label sets seen' reference.

## Decision rule

Written before any model is evaluated on the extra tasks (revised on 2026-10-11 to cover both source lists). `G` is the pooled accuracy of a model minus the pooled accuracy of the zero-shot model. `P_all` is the pool over `held_out_sources`, `P_task` the pool over `task_novel_sources`.

- **Slot supported:** on `P_all`, `G_slot - G_best_baseline` is at least 2.0 points **and** the slot model has the higher accuracy on at least 60% of the sources; **and** on `P_task` the difference is not reversed (`G_slot - G_best_baseline` is at least 0). `G_best_baseline` is the larger of the frozen-head and the trainable-head baseline.
- **Baseline supported:** the same with the roles swapped.
- **Otherwise a tie:** the simpler baseline is preferred.
- If `P_task` has fewer than 5 sources, the rule uses `P_all` alone and says so.

The 2-point margin is about two standard errors for roughly 3,000 pooled decisions. It is a proposal; the owner may change it before the evaluation.

## Limits

One seed per model. Per-source n is limited. Sources flagged by name may be familiar to Mimir (expected for option A: all of them), and only option C is unseen for both Mimir and our fine-tuning. The task-familiarity flag is by name only and can miss a renamed task. Label novelty is exact-string, not semantic.

## Status and results

Status: **planned**. Results: _not run yet_.

| Date | Run id / path | Config (overrides) | Result | Observation page |
|---|---|---|---|---|
| | | | | |
