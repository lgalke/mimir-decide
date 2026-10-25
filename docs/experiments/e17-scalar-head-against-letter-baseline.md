---
type: Experiment
title: 'E17: Scalar per-option head against the letter baseline'
description: Train the scalar head (D33) on mixture v1 and compare it with the letter baseline on the frozen evaluation, with the cost per example next to the quality.
status: done
answers: [q04, q16]
depends_on: [e16]
tags: [heads, scalar, v1]
timestamp: 2026-10-22T00:00:00Z
---

## Hypothesis

Expectation, not a result: at about 1M training examples the readouts are within noise of each other (slot and letter baseline tie). The scalar head offers no 26-option limit and the standard design, at a cost of about K forward passes per decision.

## Design change (owner, 2026-10-23): train on the v0 mixture first

To change only the head, the first run is `scalar-v0`: the scalar head on `mixture-v0` (371,275 decisions) at `max_len: 2048`, learning-rate scaling off, seed 0, effective batch 32, the same as `baseline-v0` (letter baseline, validation accuracy 0.835). What still differs besides the head: the chat-format row per option and the sampled negatives with `max_train_options: 6` (both part of the scalar readout, [D33](/design/d33-scalar-per-option-head.md), [D34](/design/d34-chat-format-for-slot-and-scalar.md)), the micro-batch (2 x 16 decisions against 8 x 4, so the batches contain different decisions), and the code version. The comparison uses `compare` on the frozen sets and the existing pre-set rules below. The v1-mixture runs (scalar on `v1-371k`, with or without dividers) follow only if this run is interesting.

```bash
# on the GPU; source .env.sh first
python -u -m mimir_decide.train --config configs/train_scalar_v1.yaml \
  --set data_dir=$MIMIR_DECIDE_DATA/mixture-v0 train_file=train.parquet max_len=2048 run_dir=runs/scalar-v0
```

**Curve so far (2026-10-23, scalar-v0 at step 4,000 of 11,603; first 2,000 validation decisions; one standard error of the accuracy is about 0.9 points):**

| Step | `baseline-v0` NLL / acc | `scalar-v0` NLL / acc |
|---|---|---|
| 1,000 | 0.597 / 0.770 | 0.626 / 0.744 |
| 2,000 | 0.584 / 0.764 | 0.604 / 0.771 |
| 3,000 | 0.550 / 0.784 | 0.569 / 0.775 |
| 4,000 | 0.536 / 0.792 | 0.539 / 0.790 |

The scalar head starts behind (random linear layer, as expected) and has caught up in NLL at step 4,000; all accuracy differences are within noise. Plot: `docs/figures/e17-scalar-v0-vs-baseline-v0.png`, made with `python -m mimir_decide.plot_curves RUN_OR_LOG ... --out FILE` (needs `matplotlib`). Intermediate values are not results; the comparison follows the rule below on the full evaluation sets.

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

Written before running. Compare with the letter baseline trained on the same subset (`train_v1-371k.parquet`), or with the 1M run if the scalar head is trained on the same data. Scalar supported: pooled E12 gain at least 2.0 points above the letter baseline on at least 60% of sources, and validation accuracy not lower by more than 1.0 point. (Corrected 2026-10-24, before any result: an earlier draft added a condition on the Typed Decisions KL; the benchmark test split must not decide between model designs, D11. It is reported, once per model with `--final`, and does not enter the rule.) Otherwise a tie; the simpler and cheaper letter baseline is preferred unless more than 26 options are needed.

## Limits

One seed; sampled negatives change the training objective (corrected, not proven equivalent); the cost per example is not measured yet; all sources are flagged as seen by Mimir.

## Status and results

Status: **done** for the first run (`scalar-v0` against `baseline-v0`): by the pre-set rule a **tie** ([O30](/observations/o30-scalar-head-ties-with-the-letter-baseline.md)): E12 gain +8.4 against +8.9 points, 10 of 32 sources better, validation accuracy 0.832 against 0.835; 51.1 h against 15.8 h. Descriptive: Typed Decisions KL 0.078 (lowest so far). The runs on mixture v1 (`v1-371k`, with dividers) are not done.

| Date | Run id / path | Config (overrides) | Result | Observation page |
|---|---|---|---|---|
| 2026-10-25 | `runs/scalar-v0` | `train_scalar_v1.yaml`, `data_dir=mixture-v0`, `max_len=2048`, 2 x 16, dividers off | validation 0.832 / NLL 0.462; E12 0.807 (gain +8.4, v0 baseline +8.9); held-out 0.965; Typed Decisions 0.784 / KL 0.078; 51.1 h, 48.7 GB; rule: tie | [O30](/observations/o30-scalar-head-ties-with-the-letter-baseline.md) |
