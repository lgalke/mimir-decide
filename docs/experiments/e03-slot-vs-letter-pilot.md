---
type: Experiment
title: 'E03: Slot readout vs letter baseline (pilot)'
description: 'The central comparison: same data, backbone, loss and calibration; only the readout differs.'
status: done
answers: [q04, q01]
depends_on: [e01, e02]
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Hypothesis

The slot readout gives lower validation and held-out NLL than the letter baseline, especially on Score decisions and with shuffled option order.

## Prerequisites

E01, E02

## Setup

```bash
# Run from the repository root with the project environment active (uv .venv or conda).
export DATA=$HOME/mimir-decide-data/mixture-v0 RUNS=$HOME/mimir-decide-data/runs
PY=python
```
```bash
SKIP_BUILD=1 DATA=$DATA RUNS=$RUNS scripts/run_pilot.sh      # slot-v0 then baseline-v0; epochs: 1 by default
$PY -m mimir_decide.compare $RUNS/slot-v0 $RUNS/baseline-v0 --file validation.json
$PY -m mimir_decide.compare $RUNS/slot-v0 $RUNS/baseline-v0 --file heldout.json
```
Repeat with `--set seed=1` (and `seed=2` if time) into `run_dir=$RUNS/slot-v0-s1` etc. Do **not** evaluate on test.

## Metrics to record

Per kind and overall: accuracy, NLL, Brier, ECE (after calibration), score MAE; pooled results for sources seen vs not flagged by Mimir. Training curves, `peak_mem_gb`, `ex_per_s`.

## Decision rule

Slot 'wins' if its NLL is lower than the baseline's on validation and on heldout for at least 2 of 3 kinds, by more than the spread between seeds. If the baseline is within the seed spread, prefer the baseline and record that D05 should be revisited. Report seen / not-flagged separately (Q01).

## Status and results

Status: **done** (one seed). The owner decided on 2026-10-09 that one seed is enough, so the 'spread between seeds' part of the decision rule cannot be applied. A fixed margin is proposed (a validation NLL gap below 0.01 is a tie, and a tie goes to the simpler baseline) and waits for the owner's confirmation. By any such margin the first result is a tie.

| Date | Run id / path | Config (overrides) | Result | Observation page |
|---|---|---|---|---|
| 2026-10-09 | `runs/slot-v0`, `runs/baseline-v0` | defaults, seed 0, via `scripts/run_pilot.sh` | tie: validation accuracy 0.831 vs 0.835, NLL 0.455 vs 0.456 | [O14](/observations/o14-first-pilot-results-slot-and-baseline-tie.md) |

When run: fill this table, create an observation page (copy the template) with the numbers, set `status` in the frontmatter, add a line to the update log, and regenerate the indexes.
