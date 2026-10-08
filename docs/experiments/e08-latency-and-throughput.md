---
type: Experiment
title: 'E08: Latency and throughput'
description: Measure decisions per second and ms per decision for the slot model against Jev's claims.
status: planned
answers: []
depends_on: [e02]
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Hypothesis

The slot model is far slower than Jev's reported 70-500 ms per call and cost grows with the number of questions, because each decision is its own forward pass.

## Prerequisites

E02 or E03 run

## Setup

```bash
# Run from the repository root with the project environment active (uv .venv or conda).
export DATA=$HOME/mimir-decide-data/mixture-v0 RUNS=runs
PY=python
```
```bash
$PY -m mimir_decide.bench --run_dir $RUNS/slot-v0 --data_dir $DATA --batch_sizes 1 8 32 --n 256
```
Writes `eval/latency.json`.

## Metrics to record

`decisions_per_s`, `ms_per_decision`, `peak_mem_gb` per batch size, on the GPU used.

## Decision rule

Informational. If latency matters, consider sharing the state encoding across questions (not implemented) or early exit.

## Status and results

Status: **planned**. Results: _not run yet_.

| Date | Run id / path | Config (overrides) | Result | Observation page |
|---|---|---|---|---|
| | | | | |

When run: fill this table, create an observation page (copy the template) with the numbers, set `status` in the frontmatter, add a line to the update log, and regenerate the indexes.
