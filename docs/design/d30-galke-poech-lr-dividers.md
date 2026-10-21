---
type: Decision
title: 'D30: Galke-Poech learning-rate dividers for the recurrent L and H stacks (flag, off by default)'
description: Option lr_cycle_dividers divides the backbone learning rate by how often each stack is applied with gradient (L by 6, H by 2 for full backprop). Off by default so that the existing runs stay reproducible.
status: accepted
date: '2026-10-21'
decided_by: owner (rule and name); implementation by agent
tags: [training, learning-rate, hrm, reproducibility]
timestamp: 2026-10-21T00:00:00Z
---

## Context

HRM-Text applies the same L stack `H_cycles x L_cycles` times (2 x 3 = 6) and the same H stack `H_cycles` times (2) per forward pass. Their weights therefore collect gradient contributions from every application. All runs so far used one backbone learning rate (1e-5) for both stacks, and one for the new head (1e-4) ([training-setup](/design/training-setup.md)). The owner asked for the 'Galke-Poech dividers': scale the base learning rate down by how often each module is applied, and make it a flag so the existing runs stay reproducible.

## Decision

- **Flag:** `lr_cycle_dividers` in the training config (`configs/train_slot.yaml`, `configs/train_baseline.yaml`), **default `false`**. With `false`, `train.py` builds exactly the parameter groups of the earlier runs (one backbone group at `lr_backbone`, weight decay 0.01; one head group at `lr_head`, weight decay 0), so the pilot, E11 and the probes remain reproducible. Override on the command line: `--set lr_cycle_dividers=true`.
- **With `true`:** the backbone is split into three groups. The L stack gets `lr_backbone / dL`, the H stack `lr_backbone / dH`, all other trainable backbone weights (input embeddings or LM head, when unfrozen) keep `lr_backbone`. The head group is unchanged.
- **Dividers:** `dH = H_cycles` (2). `dL` = the number of L-stack applications that carry gradients = the sum of `L_bp_cycles` over the high cycles (6 for H_cycles = 2, L_cycles = 3 and `L_bp_cycles = [3, 3]`, the checkpoint's setting). The owner specified 6 and 2 for full backpropagation; counting only gradient-carrying applications for truncated backprop (for example `[0, 3]` gives `dL = 3`) is my extension of that rule and is open to change.
- **Logging:** the training log gets an `lr_groups` list (L, H, other, head) next to `lr` (the first group), the first log line prints the dividers, and `decision_meta.json` stores them as `lr_dividers`. The warmup and cosine schedule scale all groups by the same factor.

## Consequences

- Adam normalises gradient scale, so the divider shrinks the step size of the stack by that factor: with the flag on, the L stack learns 6 times and the H stack 2 times slower than with the old setting at the same `lr_backbone`. If the aim is the same average step size, `lr_backbone` can be raised accordingly; this is not tuned.
- The runs so far (flag off) are not comparable with a run with the flag on without noting the flag. A comparison needs its own experiment (not planned yet).
- Interaction with [E06](/experiments/e06-l-bp-cycles-ablation.md): `L_bp_cycles = [0, 3]` changes `dL` from 6 to 3 when the flag is on.
- Tested: five unit tests (flag off equals the earlier single group; 6 and 2 for `[3, 3]`; 3 for `[0, 3]`; every trainable parameter in exactly one group; frozen backbone gives no groups) and a 3-step smoke training on a tiny random model with the flag on and off. Not run on the real model.

## Revisit when

A run with the flag on shows an effect on validation, or the owner wants different dividers (for example by applications instead of by gradient-carrying applications).

## Related

* [training-setup](/design/training-setup.md)
* [E06](/experiments/e06-l-bp-cycles-ablation.md)
* [O13](/observations/o13-l-bp-cycles-changes-gradient-scale.md)
