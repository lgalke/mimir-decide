---
type: Experiment
title: 'E04: Effect of temperature calibration'
description: Quantify ECE before and after per-kind temperature scaling, on validation and held-out tasks.
status: done
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
export DATA=$HOME/mimir-decide-data/mixture-v0 RUNS=runs
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

Note: for the pilot runs only calibrated eval files exist in git ([O17](/observations/o17-calibration-and-selective-risk-of-the-pilot-runs.md)). Run this experiment with the current code, so the eval files also contain the reliability tables, and copy the pilot checkpoints to `runs/` first (see D24).

## Metrics to record

ECE, NLL and Brier raw vs calibrated, per kind; temperature values (flag any at the clamp bounds 0.05 or 20); ECE on hard-label vs soft-label sources; reliability diagram data from selective-risk curves (`risk@coverage`).

## Decision rule

Calibration helps if ECE falls on validation and does not rise on heldout. A rise on heldout means temperatures do not transfer (Q13); record it. Do not describe results as probability of correctness (Q07).

## Status and results

Status: **done** (pilot checkpoints, one seed). Rule outcome: ECE does not fall on validation (slot 0.019 to 0.018, baseline 0.015 to 0.016) and it rises slightly on the held-out task for the slot model (0.021 to 0.027), so the temperatures do not transfer; the effect is small because the raw ECE is already small. The server code did not yet contain the reliability tables, so none were produced.

| Date | Run id / path | Config (overrides) | Result | Observation page |
|---|---|---|---|---|
| 2026-10-11 | `runs/slot-v0`, `runs/baseline-v0` | `--no_calibration --tag uncal` vs calibrated, validation and held-out | raw ECE 0.015-0.019; NLL improves by at most 0.005; noul ECE worse after calibration | [O18](/observations/o18-temperature-scaling-barely-changes-the-fine-tuned-models.md) |

When run: fill this table, create an observation page (copy the template) with the numbers, set `status` in the frontmatter, add a line to the update log, and regenerate the indexes.
