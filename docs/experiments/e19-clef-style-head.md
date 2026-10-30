---
type: Experiment
title: 'E19: Clef-style head, frozen and fully fine-tuned'
description: 'Test the Clef-style scoring head (D38): a head-only probe on mixture v1, compared with the span and letter probes, and a full fine-tuning run.'
status: planned
answers: [q04, q14]
depends_on: [e18]
tags: [heads, clef, probe, lexical-prior]
timestamp: 2026-10-31T00:00:00Z
---

## Hypothesis

Expectation, not a result. The span probe failed on unseen label sets (E12 accuracy 0.631, O31), and I suggested that a plain head over pooled option vectors learns label-string priors. The Clef-style head adds a lexical prior tied to the LM's own output embeddings and evidence routing over the whole sequence. If the hypothesis is right, the Clef probe should keep the in-distribution and benchmark strength of the span probe and avoid the loss on E12. It may also fail on both.

## Setup

Mixture v1, subset `v1-371k` (`train_file`), `max_len` 4096, chat format, effective batch 32, one epoch, seed 0, as the E18 probes.

```bash
# on the GPU job: cd /work/dfm/lukasgp/mimir-decide; git pull; source .env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. head-only probe (about 3-4 h expected); head learning rate 3e-4 (fixed in advance, not tuned)
nohup python -u -m mimir_decide.train --config configs/train_clef_v1.yaml \\
  --set freeze_backbone=true lr_head=3.0e-4 snapshot_every=2000 run_dir=runs/clef-probe-v1 > runs-clef-probe-v1.log 2>&1 &

# 2. full fine-tuning on the same subset, dividers on as in the E18 night runs (about 15 h expected)
nohup python -u -m mimir_decide.train --config configs/train_clef_v1.yaml \\
  --set lr_cycle_dividers=true ema_decay=0.999 snapshot_every=1000 gradient_checkpointing=true \\
  run_dir=runs/clef-v1-371k-lrdiv > runs-clef-v1-371k-lrdiv.log 2>&1 &
```

Evaluate as for E18 (audit once, calibrate, validation, held-out, extra, Typed Decisions `--final` once per finished model).

## Metrics to record

Validation, held-out and E12 accuracy, NLL, Brier, ECE per kind; the shares of the gap `R` for the probe; the values of `prior_logit_scale`, `joint_logit_scale` and `residual_gate` of the trained head (how much the lexical prior is used); training time and memory.

## Decision rule

Written before the runs; no selection on the test split ([D11](/design/d11-split-isolation-and-test-access.md)), the benchmark is descriptive.

- **Probe:** against `letter-probe-v1` (validation 0.701) with the E13 margin: better by at least 2 points means the head extracts more from the frozen features; within 2 points, no difference. On E12 the probe is reported against `span-probe-v1` (0.631) and `letter-probe-v1` (0.732); it supports the lexical-prior hypothesis if its E12 accuracy is within 1 point of the letter probe or higher while its validation accuracy is within 2 points of the span probe or higher.
- **Full model:** against the matched `span-v1-371k-lrdiv` and `baseline-v1-371k-lrdiv` as in [E17](/experiments/e17-scalar-head-against-letter-baseline.md): pooled E12 gain at least 2.0 points above on at least 60% of sources and validation accuracy not lower by more than 1.0 point; otherwise a tie, the letter baseline stays preferred.

## Limits

One seed per model; the head's size and learning rate are not tuned; joint decoding across questions and LoRA are not part of the implementation; our reading of the published Clef code, not Unsloth's training code.

## Status and results

Status: **planned**. Code and offline tests exist; nothing ran on the real model.

| Date | Run id / path | Config (overrides) | Result | Observation page |
|---|---|---|---|---|
| | | | | |
