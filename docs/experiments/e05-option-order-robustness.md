---
type: Experiment
title: 'E05: Option-order robustness'
description: Evaluate with options presented in random orders and measure the change in accuracy and agreement.
status: done
answers: [q12]
depends_on: [e03]
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Hypothesis

The slot model is more invariant to option order than the letter baseline.

## Prerequisites

E03 runs

## Setup

```bash
# Run from the repository root with the project environment active (uv .venv or conda).
export DATA=$HOME/mimir-decide-data/mixture-v0 RUNS=runs
PY=python
```
```bash
for R in slot-v0 baseline-v0; do for SEED in 1 2 3; do
  $PY -m mimir_decide.evaluate --run_dir $RUNS/$R --data_dir $DATA --split validation --order_seed $SEED
done; done
```
Writes `eval/validation_order<seed>.json`. Add a small snippet to compute pairwise agreement of argmax between seeds if needed.

## Metrics to record

Accuracy and NLL per seed vs identity order; spread across seeds; agreement of the predicted option across orders (per kind).

## Decision rule

Fixed on 2026-10-11, before the results were read. Let `drop` be the validation accuracy with the given option order minus the mean accuracy over the 3 random orders, per model.

- The invariance argument of [D05](/design/d05-slot-readout-heads.md) is **supported** if the baseline's `drop` is at least 1 point of accuracy larger than the slot model's `drop`.
- Otherwise it is **not supported**: the two models are equally robust to option order, and the simpler baseline is preferred.

Also report, for each model, the spread (maximum minus minimum accuracy) over the 3 orders and the agreement of the predicted option between orders. They inform the discussion but do not change the rule. The 1-point margin is a proposal of the agent that the owner did not object to; the owner can change it for later experiments.

## Status and results

Status: **done**. Rule outcome: the baseline's drop minus the slot model's drop is 0.14 points, below the 1-point margin, so the order-invariance argument is **not supported**; both models are equally robust.

| Date | Run id / path | Config (overrides) | Result | Observation page |
|---|---|---|---|---|
| 2026-10-11 | `runs/slot-v0`, `runs/baseline-v0` | `--order_seed 1,2,3` on validation | drop +0.03 points (slot), +0.17 points (baseline); spread at most 0.4 points | [O19](/observations/o19-option-order-does-not-matter-for-either-model.md) |

When run: fill this table, create an observation page (copy the template) with the numbers, set `status` in the frontmatter, add a line to the update log, and regenerate the indexes.
