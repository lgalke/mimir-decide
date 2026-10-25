---
type: Experiment
title: 'E18: Span-pooling readout with per-kind heads, frozen and fully fine-tuned'
description: A head-only probe and a full fine-tuning run of the span-pooling readout (D35) on the v0 mixture, compared with the E13 probes and with baseline-v0.
status: planned
answers: [q04, q14]
depends_on: [e13, e17]
tags: [heads, span, probe, per-kind]
timestamp: 2026-10-25T00:00:00Z
---

## Hypothesis

Expectation, not a result: reading the mean over each option's own tokens extracts much more from a frozen backbone than the marker readout of the slot probe (0.446 validation accuracy) and may reach the letter probe (0.703) or better; fully fine-tuned it ties with the other readouts (O26, O30). The owner's PhD student reports a good result with a similar readout; whether she trained only heads is open.

## Setup

Both runs use the pilot setting so that they are comparable with the E13 probes and `baseline-v0`: mixture `v0` (`train.parquet`, 371,275 decisions), `max_len` 2048, one epoch, seed 0, effective batch 32, learning-rate scaling off, chat format on, one head per kind.

```bash
# on each GPU job: cd /work/dfm/lukasgp/mimir-decide; git pull; source .env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. head-only probe (about 3-4 h, little memory); head learning rate 1e-3 as for the E13 probes
nohup python -u -m mimir_decide.train --config configs/train_span_v1.yaml \
  --set data_dir=$MIMIR_DECIDE_DATA/mixture-v0 train_file=train.parquet max_len=2048 \
  freeze_backbone=true lr_head=1.0e-3 run_dir=runs/span-probe > runs-span-probe.log 2>&1 &

# 2. full fine-tuning (about 16 h expected, like the other one-pass models)
nohup python -u -m mimir_decide.train --config configs/train_span_v1.yaml \
  --set data_dir=$MIMIR_DECIDE_DATA/mixture-v0 train_file=train.parquet max_len=2048 \
  run_dir=runs/span-v0 > runs-span-v0.log 2>&1 &
```

Evaluation as for E17 (calibrate, validation and held-out on `mixture-v0`, extra set, Typed Decisions with `--final` once per model; `--batch_size` can stay at 16).

## Metrics to record

Validation, held-out and E12 accuracy, NLL, Brier, ECE per kind; the share of the gap that the probe recovers `R = (acc_probe - 0.669) / (0.835 - 0.669)` as in E13; training time and memory.

## Decision rule

Written before the runs; no selection on the test split ([D11](/design/d11-split-isolation-and-test-access.md)); the benchmark is reported descriptively.

- **Probe:** against the letter probe (validation accuracy 0.703) with the E13 margin: better by at least 2 points means the span readout extracts more from the same frozen features; within 2 points no difference; worse by 2 points the opposite. Against the slot probe (0.446) the difference is also reported.
- **Full model:** against `baseline-v0` as in [E17](/experiments/e17-scalar-head-against-letter-baseline.md): pooled E12 gain at least 2.0 points above on at least 60% of sources, and validation accuracy not lower by more than 1.0 point; otherwise a tie, the letter baseline stays preferred.

## Limits

One seed per model; the student's method is not known in detail, so this is our reading ([D35](/design/d35-span-pooling-readout-with-per-kind-heads.md)); a shared head against per-kind heads is not tested here.

## Status and results

Status: **planned**. Code and offline tests exist; nothing ran on the real model.

| Date | Run id / path | Config (overrides) | Result | Observation page |
|---|---|---|---|---|
| | | | | |
