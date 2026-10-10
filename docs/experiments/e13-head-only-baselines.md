---
type: Experiment
title: 'E13: Head-only baselines (frozen backbone)'
description: Train only the output head on the frozen Mimir backbone, for the letter readout (head initialised from the LM head) and for the slot readout. Shows how much of the fine-tuning gain a readout alone can reach.
status: planned
answers: [q04, q14]
depends_on: [e10]
tags: [baseline, probe, frozen-backbone]
timestamp: 2026-10-11T00:00:00Z
---

## Purpose

Suggested by the owner. [E10](/experiments/e10-zero-shot-letter-baseline.md) showed that fine-tuning lifts accuracy from 0.669 to 0.835. Two questions remain: how much of that gain needs the backbone to change, and does the slot readout extract more than the letter readout from the same representations?

## Variants

Both use the original mixture, the same loss and data as the pilot, a frozen backbone and frozen input embeddings.

- **Letter probe.** The classifier head is a trainable copy of the 26 letter rows of the language-model head (`train_letter_head=true`). Step 0 is exactly the zero-shot model, so the curve starts at the E10 result.
- **Slot probe.** The head is the slot head (LayerNorm, linear layer, 3 scales; about 5,000 parameters). The marker vector stays at its initialisation (the mean embedding), because training it would need a backward pass through the backbone.

Nothing in the backbone changes, so its weights are not stored; the checkpoint holds only the head and loads the base model from the hub, like the zero-shot checkpoint. There is no backward pass through the backbone, so these runs need much less memory and time than a full fine-tuning run. The time is not measured. Expect several hours, not 17; the validation curve is logged every 1,000 steps, so a run can be stopped when it flattens.

## Setup

```bash
# Run from the repository root with the project environment active (uv .venv or conda).
export DATA=$HOME/mimir-decide-data/mixture-v0
PY=python
COMMON="data_dir=$DATA freeze_backbone=true lr_head=1.0e-3"     # use the pilot's other overrides if there were any

$PY -m mimir_decide.train --config configs/train_baseline.yaml --set $COMMON train_letter_head=true run_dir=runs/letter-probe
$PY -m mimir_decide.train --config configs/train_slot.yaml     --set $COMMON run_dir=runs/slot-probe

for R in letter-probe slot-probe; do
  $PY -m mimir_decide.calibrate --run_dir runs/$R --data_dir $DATA
  for S in validation heldout; do $PY -m mimir_decide.evaluate --run_dir runs/$R --data_dir $DATA --split $S; done
done
$PY -m mimir_decide.compare runs/letter-zeroshot runs/letter-probe runs/slot-probe runs/baseline-v0 runs/slot-v0 --file validation.json
```

The head learning rate 1e-3 is fixed here, before the runs: with a frozen backbone, a head-only model usually needs a larger rate than the 1e-4 of the full fine-tuning. It is not tuned. Later the same models can be tested on the extra tasks of [E12](/experiments/e12-unseen-label-sets.md).

## Metrics to record

Validation and held-out accuracy, NLL, Brier, ECE and confidence, per kind, with the seen-by-Mimir split. The validation curve in the training log. The share of the gap that the head recovers: `R = (acc_probe - 0.669) / (0.835 - 0.669)`, with 0.669 from the zero-shot model and 0.835 from the fine-tuned baseline.

## Interpretation rule

Written before the runs. These are diagnostics, they do not select a model.

- **Letter probe.** `R` below 0.25: the fine-tuning gain comes mostly from the backbone, not from the readout. `R` above 0.75: mostly from readout and format adaptation. In between: both contribute.
- **Slot probe against letter probe on the same frozen backbone.** The slot probe is better by at least 2 points of accuracy: the slot readout extracts more from the same representations, which supports [D05](/design/d05-slot-readout-heads.md). Within 2 points: no difference. The letter probe is better by at least 2 points: the opposite.

My expectation, not a result: the letter probe gains little over zero-shot, because it can only re-weight 26 fixed directions of a hidden state that the frozen backbone computed for next-token prediction. The slot probe is the real unknown, because its marker positions have seen the whole prompt through the bidirectional attention but the backbone was never trained to make them informative.

## Limits

One seed. The marker vector of the slot probe is not trained, which handicaps it compared with the slot model. The head learning rate is not tuned; a probe with a poorly chosen rate could look weaker than it is. The frozen backbone is evaluated in bf16, as in the other runs.

## Status and results

Status: **planned**. Results: _not run yet_. The code is implemented and tested on a tiny model (`freeze_backbone`, `train_letter_head`).

| Date | Run id / path | Config (overrides) | Result | Observation page |
|---|---|---|---|---|
| | | | | |
