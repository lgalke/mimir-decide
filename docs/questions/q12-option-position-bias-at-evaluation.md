---
type: Question
title: 'Q12: How position-sensitive are the predictions?'
description: Evaluation uses the given option order; a model could still prefer early or late slots.
status: answered
priority: medium
resolved_by: [e05]
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Why it matters

Training shuffles order ([D05](/design/d05-slot-readout-heads.md)) but this is untested.

## What we know

`evaluate.py --order_seed N` presents options in a seeded random order and maps logits back.

## How to resolve

[E05](/experiments/e05-option-order-robustness.md): compare identity vs several seeds, per model.

## Answer

Not measurably. With 3 random option orders, validation accuracy changes by 0.03 points (slot) and 0.17 points (baseline) against the given order, with spreads of at most 0.4 points ([O19](/observations/o19-option-order-does-not-matter-for-either-model.md)). Agreement of the predicted option between orders was not computed (the eval files hold no per-decision predictions).

## Decision impact

Reliability of decisions when a caller orders options arbitrarily.
