---
type: Experiment
title: 'E02: Smoke run on the real model'
description: Load Mimir v1.5, train 50 steps, measure memory and throughput, confirm loss decreases and checkpoints reload.
status: planned
answers: [q03]
depends_on: [e01]
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Hypothesis

Mimir v1.5 slot training fits on one GPU at `max_len 2048`, batch 8, with gradient checkpointing, and the loss falls within 50 steps.

## Prerequisites

E01 (or any mixture with a few thousand rows)

## Setup

```bash
export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/mimir-decide
export DATA=$HOME/mimir-decide-data/mixture-v0 RUNS=$HOME/mimir-decide-data/runs
PY=$UV_PROJECT_ENVIRONMENT/bin/python
```
```bash
$PY -m mimir_decide.train --config configs/train_slot.yaml --set data_dir=$DATA run_dir=$RUNS/smoke-slot max_steps=50 eval_every=25 eval_max_examples=200 log_every=5
$PY -m mimir_decide.calibrate --run_dir $RUNS/smoke-slot --data_dir $DATA
$PY -m mimir_decide.evaluate --run_dir $RUNS/smoke-slot --data_dir $DATA --split validation
```
If it runs out of memory: add `batch_size=4 grad_accum=8`, then `max_len=1024`, then `L_bp_cycles=[0,3]`, then `param_dtype=bfloat16`, and record which one was needed. Repeat once with `configs/train_baseline.yaml`.

## Metrics to record

`peak_mem_gb`, `ex_per_s` and `loss` in `train_log.jsonl`; `skipped` count; time to load; whether `best/` reloads in `calibrate.py` (it uses `load_model`); first validation NLL vs the uniform baseline `mean(ln K)`.

## Decision rule

Pass: no OOM at the config used, loss lower at step 50 than at step 1, evaluate runs. Record the working config as an observation and update D04/D09/D10 and Q03.

## Status and results

Status: **planned**. Results: _not run yet_.

| Date | Run id / path | Config (overrides) | Result | Observation page |
|---|---|---|---|---|
| | | | | |

When run: fill this table, create an observation page (copy the template) with the numbers, set `status` in the frontmatter, add a line to the update log, and regenerate the indexes.
