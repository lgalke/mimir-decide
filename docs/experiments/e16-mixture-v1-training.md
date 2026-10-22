---
type: Experiment
title: 'E16: Training on mixture v1 (composition, size and 4096 context)'
description: Train the slot model and the letter baseline on the larger, more diverse mixture v1, first at v0's size, then larger, and compare on the frozen evaluation.
status: running
answers: [q16, q01]
depends_on: [e12, e14]
tags: [mixture, diversity, context-4096, v1]
timestamp: 2026-10-21T00:00:00Z
---

## Hypothesis

A mixture with more sources, balanced kinds and fewer binary decisions ([D31](/design/d31-mixture-v1-size-and-diversity.md)) improves transfer to unseen label sets ([E12](/experiments/e12-unseen-label-sets.md)) and the Typed Decisions benchmark more than the same number of decisions drawn as in v0. A larger set helps further. Expectation, not a result.

## Confounds

v1 changes composition, size and (with the owner's plan) the context length from 2048 to 4096 at the same time. The nested design separates them: `v1-371k` has the size of v0, and an evaluation of the existing v0 models at 4096 costs no training. A v0-data run at 4096 is the full control and optional.

## Prerequisites

`mixture-v1` built on the server with `configs/data_v1.yaml`; the diversity report and the eval-freeze check recorded ([D31](/design/d31-mixture-v1-size-and-diversity.md)); a 50-step smoke run of the real model at `max_len: 4096` (memory and examples per second on the B200).

## Setup

```bash
# Run from the repository root with the project environment active; MIMIR_DECIDE_DATA set (source .env.sh).
export DATA=$MIMIR_DECIDE_DATA/mixture-v1

# 1. Build (hours, network; no second build into the same folder) and look at the result
python -m mimir_decide.build_mixture --config configs/data_v1.yaml 2>&1 | tee $DATA.build.log
python -m mimir_decide.mixture_report --data_dir $DATA --compare $MIMIR_DECIDE_DATA/mixture-v0
python -m mimir_decide.mixture_report --data_dir $DATA --file train_v1-371k.parquet

# 2. Smoke run at 4096 (50 steps), then the two models on v1-371k
python -m mimir_decide.train --config configs/train_slot_v1.yaml --set max_steps=50 run_dir=runs/smoke-v1
python -m mimir_decide.train --config configs/train_slot_v1.yaml
python -m mimir_decide.train --config configs/train_baseline_v1.yaml

# 3. Calibrate and evaluate on the frozen sets (validation, heldout, extra) and once on the benchmark test
for R in slot-v1-371k baseline-v1-371k; do
  python -m mimir_decide.calibrate --run_dir runs/$R --data_dir $DATA
  for S in validation heldout; do python -m mimir_decide.evaluate --run_dir runs/$R --data_dir $DATA --split $S; done
  python -m mimir_decide.evaluate --run_dir runs/$R --data_dir $MIMIR_DECIDE_DATA/mixture-extra-eval --split extra
  python -m mimir_decide.typed_decisions --run_dir runs/$R --final
done
```

The larger run (`--set train_file=train_v1-1M.parquet run_dir=runs/slot-v1-1M`, or `train.parquet` for the full set) is chosen after the throughput is known. `validation` and `calib` come from `mixture-v1`, which equals `mixture-v0`.

## Fast path (owner's priority, 2026-10-21)

The owner wants a good model soon and accepts weaker attribution. Build the 1M mixture directly (`--set sampling.total=1000000 per_source_cap=12000`; `train.parquet` is then the 1M set and `train_v1-371k.parquet` comes for free) and train one model on it at `max_len: 4096`, the letter baseline first (the simpler model under the tie rules; the slot model can follow on the same data). The composition-only run on `v1-371k` stays available later; the decision rule below then applies only to what was actually run.

## Metrics to record

Validation, heldout, E12 pooled accuracy and gain over zero-shot (both source lists), Typed Decisions accuracy, KL, Brier (calibrated), ECE; the diversity report of every mixture used; examples per second and peak memory at 4096; sequence-length statistics (rows cut by the formatter).

## Decision rule

Written before running. No model selection on the test splits.

- **Composition (v1-371k against v0, same size):** better if validation accuracy rises by at least 1.0 point **and** the E12 pooled gain over zero-shot rises by at least 1.0 point, for the model in question. One of the two: partly better, reported as such. Neither: a tie.
- **Size (v1-1M or larger against v1-371k):** same two conditions.
- **Slot against baseline on v1:** the tie rule of [E12](/experiments/e12-unseen-label-sets.md) (2 points of pooled gain, 60% of sources).
- The 4096 context is reported as its own effect with the v0 models evaluated at 4096.

## Limits

One seed per model. Validation is in-distribution for the mixture; all sources are flagged as seen by Mimir; the E12 sources are all bekko. The benchmark share of LocalLLaMA in training changes with the mixture (check the report), which matters for the Typed Decisions comparison.

## Status and results

Status: **running**. The 1M mixture is built ([O27](/observations/o27-mixture-v1-build-results.md)): 999,906 decisions, the same 366 sources as v0, evaluation identical, noul 19.6%, score 11.6% and binary 43.2% (two targets missed). Training on it has not been reported yet.

| Date | Run id / path | Config (overrides) | Result | Observation page |
|---|---|---|---|---|
| 2026-10-22 | `mixture-v1` (server) | `configs/data_v1.yaml`, `sampling.total=1000000 per_source_cap=12000` | 999,906 train decisions, 366 sources, frozen evaluation identical | [O27](/observations/o27-mixture-v1-build-results.md) |
