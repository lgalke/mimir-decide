---
type: Observation
title: 'O09: Toy task: slot learns shuffled options quickly, letter baseline does not'
description: 1-layer tiny HRM, 3 options, label from a keyword in the state.
date: '2026-10-04'
confidence: low
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Observation

Slot model: loss 1.1 to about a third within 120 steps with shuffled options. Letter baseline: loss 1.12 to 0.81 after 400 steps with shuffled options; 1.09 to 0.41 in 150 steps with fixed option order.

## Interpretation

Mechanism check only. The real model is instruction-tuned with lettered options and may behave very differently ([Q04](/questions/q04-slot-vs-letter-on-real-model.md)).

## Reproduce

`tests/test_model_and_metrics.py` (slot) and the diagnostic in the log; synthetic data `synth()`.

## Related

* [heads-and-baseline](/design/heads-and-baseline.md)
