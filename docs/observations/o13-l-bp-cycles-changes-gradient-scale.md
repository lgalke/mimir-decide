---
type: Observation
title: 'O13: L_bp_cycles [0,3] shrinks L-stack gradients about 230x'
description: 'Single-batch probe on the tiny model: the L stack gets far smaller gradients with [0,3] than with the checkpoint''s [3,3].'
date: '2026-10-04'
confidence: low
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Observation

Probe on the tiny random HRM (same code path as the real model), one batch, identical weights and seed: L-stack gradient norm 0.242 with `L_bp_cycles [3,3]` and 0.00105 with `[0,3]`. With `[0,3]` the first H cycle runs its L cycles under `no_grad`, and only the last L cycle of the second H cycle receives gradients. The override takes effect (`config.L_bp_cycles` and the padded list used in the forward pass both change).

Smoke-run validation NLL was identical (1.1299) for default, gradient-checkpointing, worker and `[0,3]` runs: the random tiny model barely moves in 20 steps, so those equalities carry no information.

## Interpretation

`[0,3]` is not just a memory saving; it changes how the backbone is trained. This is a toy-model, single-batch probe: do not extrapolate magnitudes to the real model. It motivates measuring quality, not only memory, in [E06](/experiments/e06-l-bp-cycles-ablation.md).

## Reproduce

Build the model twice with `build_model` (`L_bp_cycles: null` and `[0,3]`), run `decision_loss(...).backward()` on one packed batch, and compare `model.lm.model.L_module` gradient norms.

## Related

* [D09](/design/d09-keep-checkpoint-l-bp-cycles.md)
* [Q03](/questions/q03-real-model-memory-and-speed.md)
