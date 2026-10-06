---
type: Experiment
title: 'E04: Effect of temperature calibration'
description: Quantify ECE before and after per-kind temperature scaling, on validation and held-out tasks.
status: planned
answers: [q07, q13]
depends_on: [e03]
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Hypothesis

Per-kind temperatures reduce ECE on validation and transfer, at least partly, to held-out tasks.

## Prerequisites

E03 runs

## Setup

```bash
# Run from the repository root with the project environment active (uv .venv or conda).
export DATA=$HOME/mimir-decide-data/mixture-v0 RUNS=$HOME/mimir-decide-data/runs
PY=python
```
```bash
for R in slot-v0 baseline-v0; do
  for S in validation heldout; do
    $PY -m mimir_decide.evaluate --run_dir $RUNS/$R --data_dir $DATA --split $S --no_calibration   # overwrites eval/$S.json: copy first
    $PY -m mimir_decide.evaluate --run_dir $RUNS/$R --data_dir $DATA --split $S
  done
done
```
Copy `eval/<split>.json` to `eval/<split>_uncal.json` between the two calls. Also read `calibration.json` for temperatures and the `calib_ece_before/after` fields.

## Metrics to record

ECE, NLL and Brier raw vs calibrated, per kind; temperature values (flag any at the clamp bounds 0.05 or 20); ECE on hard-label vs soft-label sources; reliability diagram data from selective-risk curves (`risk@coverage`).

## Decision rule

Calibration helps if ECE falls on validation and does not rise on heldout. A rise on heldout means temperatures do not transfer (Q13); record it. Do not describe results as probability of correctness (Q07).

## Status and results

Status: **planned**. Results: _not run yet_.

| Date | Run id / path | Config (overrides) | Result | Observation page |
|---|---|---|---|---|
| | | | | |

When run: fill this table, create an observation page (copy the template) with the numbers, set `status` in the frontmatter, add a line to the update log, and regenerate the indexes.
