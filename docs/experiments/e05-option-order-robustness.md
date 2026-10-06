---
type: Experiment
title: 'E05: Option-order robustness'
description: Evaluate with options presented in random orders and measure the change in accuracy and agreement.
status: planned
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
export DATA=$HOME/mimir-decide-data/mixture-v0 RUNS=$HOME/mimir-decide-data/runs
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

Report spread per model. If the slot model's spread is not clearly smaller, D05's invariance argument is not supported.

## Status and results

Status: **planned**. Results: _not run yet_.

| Date | Run id / path | Config (overrides) | Result | Observation page |
|---|---|---|---|---|
| | | | | |

When run: fill this table, create an observation page (copy the template) with the numbers, set `status` in the frontmatter, add a line to the update log, and regenerate the indexes.
