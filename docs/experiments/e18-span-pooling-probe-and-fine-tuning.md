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

## Setup (changed 2026-10-25 on the owner's request: both runs on mixture v1)

The two runs are not an exact replication of the PhD student's method (her backbone, trained parameters, data and prompt are unknown); they cover the two readings of it: heads only, and the whole model. Mixture v1 at `max_len` 4096, chat format, one head per kind, effective batch 32, one epoch, seed 0, learning-rate scaling off.

- **Probe** (span readout, frozen backbone) on `train_v1-371k.parquet` (the heads need few decisions; about 4-5 h expected), followed by the **letter probe on the same data** as the matched reference (new: E13 had them on v0 only).
- **Full fine-tuning** on the 1M set (`train.parquet`), to be compared with `baseline-v1-1M` (letter baseline, same data, context and settings; 39.3 h). About 39 h expected, like the letter model (one pass).

```bash
# GPU 1: both probes one after the other (the head learning rate 1e-3 as for the E13 probes)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
(python -u -m mimir_decide.train --config configs/train_span_v1.yaml \
   --set freeze_backbone=true lr_head=1.0e-3 run_dir=runs/span-probe-v1 > runs-span-probe-v1.log 2>&1 && \
 python -u -m mimir_decide.train --config configs/train_baseline_v1.yaml \
   --set freeze_backbone=true train_letter_head=true lr_head=1.0e-3 run_dir=runs/letter-probe-v1 > runs-letter-probe-v1.log 2>&1) &

# GPU 2: full fine-tuning on the 1M set
nohup python -u -m mimir_decide.train --config configs/train_span_v1.yaml \
  --set train_file=train.parquet gradient_checkpointing=true run_dir=runs/span-v1-1M > runs-span-v1-1M.log 2>&1 &
```

Before evaluating any v1 run, run `python -m mimir_decide.audit_mimir_overlap --data_dir $MIMIR_DECIDE_DATA/mixture-v1` once (it writes the list of sources flagged as seen by Mimir that the evaluation files carry; it was missing for the 1M letter run, O29). The evaluation is the same as for E17 with `--data_dir $MIMIR_DECIDE_DATA/mixture-v1`. The earlier plan on the v0 mixture (probe and full run at 2048, directly comparable with the E13 probes and `baseline-v0`) stays possible as a second step.

## Metrics to record

Validation, held-out and E12 accuracy, NLL, Brier, ECE per kind; the share of the gap that the probe recovers `R = (acc_probe - 0.669) / (0.835 - 0.669)` as in E13; training time and memory.

## Decision rule

Written before the runs; no selection on the test split ([D11](/design/d11-split-isolation-and-test-access.md)); the benchmark is reported descriptively.

- **Probe:** against the letter probe on the same data (`letter-probe-v1`; on v0 it reached validation accuracy 0.703) with the E13 margin: better by at least 2 points means the span readout extracts more from the same frozen features; within 2 points no difference; worse by 2 points the opposite. Against the slot probe (0.446) the difference is also reported.
- **Full model:** against `baseline-v1-1M` (validation accuracy 0.845, E12 accuracy 0.795) as in [E17](/experiments/e17-scalar-head-against-letter-baseline.md): pooled E12 gain at least 2.0 points above on at least 60% of sources, and validation accuracy not lower by more than 1.0 point; otherwise a tie, the letter baseline stays preferred.

## Limits

One seed per model; the student's method is not known in detail, so this is our reading ([D35](/design/d35-span-pooling-readout-with-per-kind-heads.md)); a shared head against per-kind heads is not tested here.

## Status and results

Status: **planned**. Code and offline tests exist; nothing ran on the real model.

| Date | Run id / path | Config (overrides) | Result | Observation page |
|---|---|---|---|---|
| | | | | |
