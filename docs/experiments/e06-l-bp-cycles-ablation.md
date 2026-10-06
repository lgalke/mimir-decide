---
type: Experiment
title: 'E06: L_bp_cycles ablation'
description: Compare `[3,3]` (checkpoint) with `[0,3]` for memory, speed and quality.
status: planned
answers: [q03, q10]
depends_on: [e02]
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Hypothesis

`[0,3]` saves substantial memory and time with little quality loss.

Note: a toy probe found `[0,3]` gives the L stack ~230x smaller gradients ([O13](/observations/o13-l-bp-cycles-changes-gradient-scale.md)), so also compare learning-rate sensitivity (try `lr_backbone` 2e-5 with `[0,3]`).

## Prerequisites

E02

## Setup

```bash
# Run from the repository root with the project environment active (uv .venv or conda).
export DATA=$HOME/mimir-decide-data/mixture-v0 RUNS=$HOME/mimir-decide-data/runs
PY=python
```
```bash
$PY -m mimir_decide.train --config configs/train_slot.yaml --set data_dir=$DATA run_dir=$RUNS/slot-lbp03 'L_bp_cycles=[0,3]'
```
Compare with `slot-v0` (default `[3,3]`) at identical seed, data, steps.

## Metrics to record

`peak_mem_gb`, `ex_per_s`, final validation NLL and heldout NLL.

## Decision rule

Adopt `[0,3]` as default only if validation NLL is within seed spread and the saving is material; otherwise keep D09.

## Status and results

Status: **planned**. Results: _not run yet_.

| Date | Run id / path | Config (overrides) | Result | Observation page |
|---|---|---|---|---|
| | | | | |

When run: fill this table, create an observation page (copy the template) with the numbers, set `status` in the frontmatter, add a line to the update log, and regenerate the indexes.
