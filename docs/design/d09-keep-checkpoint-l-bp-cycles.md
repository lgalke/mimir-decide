---
type: Decision
title: 'D09: Keep the checkpoint''s L_bp_cycles [3,3]'
description: Fine-tune with the same gradient routing the checkpoint was trained with.
status: accepted
date: '2026-10-04'
decided_by: agent
tags: [training, memory]
timestamp: '2026-10-04T00:00:00Z'
---

## Context

v1 trained with `[0,3]`, v1.5 with `[3,3]` (gradients through all L cycles of both H cycles). `[0,3]` is cheaper.

## Decision

Default `L_bp_cycles: null` keeps `[3,3]`. Override with `--set L_bp_cycles=[0,3]`.

## Consequences

Higher activation memory and compute. Gradient checkpointing is on by default. `[0,3]` is not only cheaper: on a toy probe it cut L-stack gradients about 230x ([O13](/observations/o13-l-bp-cycles-changes-gradient-scale.md)). Effect on quality is untested ([E06](/experiments/e06-l-bp-cycles-ablation.md)).

## Update 2026-10-09

Measured: `[3,3]` with the defaults fits at a peak of 33.6 GB on an NVIDIA RTX PRO 6000 Blackwell Server Edition ([O15](/observations/o15-pilot-training-memory-and-speed.md)). The decision stands.

## Revisit when

[E02](/experiments/e02-smoke-run-real-model.md) shows `[3,3]` does not fit.

## Related

* [mimir-v1-5](/research/mimir-v1-5.md)
* [hrm-text](/research/hrm-text.md)
