---
type: Decision
title: 'D18: Loss: soft cross-entropy plus EMD for scores'
description: Cross-entropy against the target distribution for all kinds; squared-CDF penalty (weight 0.5) for Score.
status: accepted
date: '2026-10-04'
decided_by: agent
tags: [training, loss]
timestamp: '2026-10-04T00:00:00Z'
---

## Context

Targets are partly hard labels and partly soft distributions; the colleague advised cross-entropy for hard labels and distributional cross-entropy for reliable soft targets.

## Decision

`decision_loss`: `-sum(t log p)` for every row, plus `0.5 * sum((cdf_p - cdf_t)^2)` on score rows. The EMD term is invariant to reversing the option order, which is why score options may be presented reversed.

## Consequences

Soft targets are not always reliable probabilities of correctness ([Q07](/questions/q07-soft-label-semantics-vs-correctness.md)); there is no per-source weighting yet.

## Revisit when

Some sources dominate or have noisy soft labels (add weights, or use hard labels for them).

## Related

* [heads-and-baseline](/design/heads-and-baseline.md)
* [calibration-training](/research/calibration-training.md)
