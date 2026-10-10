---
type: Decision
title: 'D26: Evaluate unseen label sets on additional tasks, without retraining'
description: Test transfer to unseen label sets by evaluating the existing models on extra tasks that are not in the training mixture, instead of retraining on a reduced mixture.
status: proposed
date: '2026-10-11'
decided_by: owner (suggested the approach); design by agent, pending the owner's choice of task sources
tags: [evaluation, generalisation, comparability]
timestamp: 2026-10-11T00:00:00Z
---

## Context

[D25](/design/d25-leave-tasks-out-for-unseen-label-sets.md) held out whole tasks and retrained both models on the rest. The owner pointed out that this makes the runs incomparable with the pilot and with the stronger baseline ([E11](/experiments/e11-stronger-baseline-trainable-lm-head.md)), and suggested adding tasks with unseen label sets instead.

## Decision

- [E12](/experiments/e12-unseen-label-sets.md) evaluates the zero-shot model, `slot-v0`, `baseline-v0` and the trainable-head baseline of E11 on the same extra tasks. Nothing is retrained.
- E11 runs first, on the original mixture, with the pilot's settings, so that it stays comparable with the pilot.
- Candidate tasks must pass the measured label-novelty rule and the leakage guards against the training data.
- Open choice for the owner: where the extra tasks come from. (A) permissive sources that the strict licence review kept out of training, used for evaluation only, which relaxes D02 and D13 from 'train and evaluate' to 'train'; (B) other permissive datasets, each needing a converter; (C) newly written or post-cutoff tasks, the only ones unseen by Mimir as well.

## Consequences

- E12 costs minutes of evaluation after E11, instead of 33 GPU hours.
- Tasks from A are mostly seen by Mimir; the clean test needs C.
- A new tool is needed to build the extra evaluation file. `select_heldout` scoring is reused; `derive_mixture` becomes a fallback.

## Revisit when

The owner chooses the sources, or a design of the leave-tasks-out kind becomes necessary (for example if too few extra tasks exist).

## Related

* [D25](/design/d25-leave-tasks-out-for-unseen-label-sets.md)
* [D02](/design/d02-permissive-licences-only.md)
* [D13](/design/d13-bekko-licence-and-test-handling.md)
