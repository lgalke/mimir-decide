---
type: Question
title: 'Q12: How position-sensitive are the predictions?'
description: Evaluation uses the given option order; a model could still prefer early or late slots.
status: open
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

## Decision impact

Reliability of decisions when a caller orders options arbitrarily.
