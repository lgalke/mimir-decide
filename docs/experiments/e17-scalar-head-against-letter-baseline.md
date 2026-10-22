---
type: Experiment
title: 'E17: Scalar per-option head against the letter baseline'
description: Train the scalar head (D33) on mixture v1 and compare it with the letter baseline on the frozen evaluation, with the cost per example next to the quality.
status: planned
answers: [q04, q16]
depends_on: [e16]
tags: [heads, scalar, v1]
timestamp: 2026-10-22T00:00:00Z
---

## Hypothesis

Expectation, not a result: at about 1M training examples the readouts are within noise of each other (slot and letter baseline tie). The scalar head offers no 26-option limit and the standard design, at a cost of about K forward passes per decision.

## Prerequisites

The letter baseline on mixture v1 is trained or running (E16); the GPU is free for the smoke test; `configs/train_scalar_v1.yaml` ([D33](/design/d33-scalar-per-option-head.md)).

## Setup

```bash
# Run from the repository root with the project environment active; MIMIR_DECIDE_DATA set (source .env.sh).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. Memory and speed: rows per decision are K, so test 2 and 4 decisions with 6 options per decision
python -u -m mimir_decide.mem_probe --config configs/train_scalar_v1.yaml --options 6 \
  --batch_sizes 1 2 4 --checkpointing true

# 2. Smoke run, then the run on the 371k subset (cheaper by 2.7x than the 1M set)
python -u -m mimir_decide.train --config configs/train_scalar_v1.yaml --set max_steps=50 run_dir=runs/smoke-scalar
python -u -m mimir_decide.train --config configs/train_scalar_v1.yaml

# 3. Calibrate and evaluate on the frozen sets (as for E16)
R=scalar-v1-371k; DATA=$MIMIR_DECIDE_DATA/mixture-v1
python -m mimir_decide.calibrate --run_dir runs/$R --data_dir $DATA
for S in validation heldout; do python -m mimir_decide.evaluate --run_dir runs/$R --data_dir $DATA --split $S; done
python -m mimir_decide.evaluate --run_dir runs/$R --data_dir $MIMIR_DECIDE_DATA/mixture-extra-eval --split extra
python -m mimir_decide.typed_decisions --run_dir runs/$R --final
```

## Metrics to record

Validation, heldout, E12 pooled gain over zero-shot (both lists), Typed Decisions accuracy, KL and Brier, ECE; examples per second and peak memory; the rows per decision actually seen.

## Decision rule

Written before running. Compare with the letter baseline trained on the same subset (`train_v1-371k.parquet`), or with the 1M run if the scalar head is trained on the same data. Scalar supported: pooled E12 gain at least 2.0 points above the letter baseline on at least 60% of sources, and the Typed Decisions KL not worse. Otherwise a tie; the simpler and cheaper letter baseline is preferred unless more than 26 options are needed.

## Limits

One seed; sampled negatives change the training objective (corrected, not proven equivalent); the cost per example is not measured yet; all sources are flagged as seen by Mimir.

## Status and results

Status: **planned**. Code and offline tests exist; nothing ran on the GPU.

| Date | Run id / path | Config (overrides) | Result | Observation page |
|---|---|---|---|---|
| | | | | |
