---
type: Experiment
title: 'E11: Stronger baseline with a trainable language-model head'
description: Repeat the letter baseline with the LM head unfrozen, so the baseline is not handicapped before it is compared with the slot model.
status: planned
answers: [q04]
depends_on: [e10]
tags: [baseline, lm-head]
timestamp: 2026-10-09T00:00:00Z
---

## Hypothesis

In the first pilot the baseline had a frozen language-model head ([D06](/design/d06-letter-logit-baseline.md)). With the head trainable, the baseline can also adapt the output rows of the letter tokens. It then matches or beats the frozen baseline. If it also matches the slot model, the simpler baseline is the better design.

## Prerequisites

[E10](/experiments/e10-zero-shot-letter-baseline.md) first (owner's order, done). Now the next experiment (owner's order on 2026-10-11). It runs on the original mixture with the pilot's settings, so it stays comparable with the pilot. [E12](/experiments/e12-unseen-label-sets.md) then evaluates it together with the other models on extra tasks. The mixture from E01 and the same settings as the pilot, so the comparison stays fair.

## Setup

```bash
# Run from the repository root with the project environment active (uv .venv or conda).
export DATA=$HOME/mimir-decide-data/mixture-v0 RUNS=runs
PY=python

$PY -m mimir_decide.train --config configs/train_baseline.yaml --set data_dir=$DATA run_dir=$RUNS/baseline-lmhead train_lm_head=true
$PY -m mimir_decide.calibrate --run_dir $RUNS/baseline-lmhead --data_dir $DATA
for S in validation heldout; do $PY -m mimir_decide.evaluate --run_dir $RUNS/baseline-lmhead --data_dir $DATA --split $S; done
$PY -m mimir_decide.compare $RUNS/baseline-lmhead $RUNS/baseline-v0 $RUNS/slot-v0 --file validation.json
```

The run goes to `runs/baseline-lmhead/`. Do not change any other setting: same seed, same data, same schedule as `baseline-v0`. If the pilot used overrides (`EXTRA_SET`), pass the same ones.

Design note: the LM head is pretrained, so its rows train at the backbone learning rate (1e-5), not at the 1e-4 of the slot model's new head. Only the 26 letter rows receive gradient, because the readout multiplies only those rows. A variant with a higher rate for those rows is possible but not planned.

Cost notes: the LM head has about 0.40B parameters (262,144 x 1,536), so gradients and AdamW state add roughly 5 GB to the 33.6 GB peak of the pilot ([O15](/observations/o15-pilot-training-memory-and-speed.md)). Only the 26 letter rows receive gradient, but weight decay touches all rows (a negligible shrink). Expect a run time similar to the pilot (about 17 hours).

## Metrics to record

As in [E03](/experiments/e03-slot-vs-letter-pilot.md): accuracy, NLL, Brier, ECE per kind, seen versus not-flagged, held-out. Also the training memory and speed.

## Decision rule

Written before running. A fixed margin replaces the seed spread, because the owner chose one seed. A difference in validation NLL below 0.01 is a tie, and in a tie the simpler model (no extra head) is preferred. This margin is a proposal until the owner confirms it.

## Status and results

Status: **planned**. Results: _not run yet_.

| Date | Run id / path | Config (overrides) | Result | Observation page |
|---|---|---|---|---|
| | | | | |
