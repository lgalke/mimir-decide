---
type: Decision
title: 'D26: Evaluate unseen label sets on additional tasks, without retraining'
description: Test transfer to unseen label sets by evaluating the existing models on extra tasks that are not in the training mixture, instead of retraining on a reduced mixture.
status: accepted
date: '2026-10-11'
decided_by: owner (suggested the approach and chose option A on 2026-10-11); design by agent
tags: [evaluation, generalisation, comparability]
timestamp: 2026-10-11T00:00:00Z
---

## Context

[D25](/design/d25-leave-tasks-out-for-unseen-label-sets.md) held out whole tasks and retrained both models on the rest. The owner pointed out that this makes the runs incomparable with the pilot and with the stronger baseline ([E11](/experiments/e11-stronger-baseline-trainable-lm-head.md)), and suggested adding tasks with unseen label sets instead.

## Decision

- [E12](/experiments/e12-unseen-label-sets.md) evaluates the zero-shot model, `slot-v0`, `baseline-v0` and the trainable-head baseline of E11 on the same extra tasks. Nothing is retrained.
- E11 runs first, on the original mixture, with the pilot's settings, so that it stays comparable with the pilot.
- Candidate tasks must pass the measured label-novelty rule and the leakage guards against the training data.
- **Source of the extra tasks: option A, chosen by the owner on 2026-10-11.** Permissive sources that the strict licence review kept out of the training mixture are used for **evaluation only**. This relaxes [D02](/design/d02-permissive-licences-only.md) and [D13](/design/d13-bekko-licence-and-test-handling.md) from 'train and evaluate' to 'train' for these sources, with these limits:
  - allowed: bekko subsets whose upstream licences pass the allowlist but whose review status is 'qualified', and allowlisted tasksource sources that are not in the training mixture;
  - not allowed: unknown, non-commercial or not-allowlisted licences, as before;
  - the rows are never used for training, calibration or model selection, and they are checked against the training data with the leakage guards;
  - the excluded-source list and the licence of every source are written into the output, so the use can be audited.
- Options B (other permissive datasets, each needing a converter) and C (newly written or post-cutoff tasks, the only ones unseen by Mimir as well) stay possible additions; they are not decided.

## Consequences

- E12 costs minutes of evaluation after E11, instead of 33 GPU hours.
- Tasks from A are mostly seen by Mimir; the clean test needs C, which stays open ([Q01](/questions/q01-clean-generalisation-evidence.md)).
- A new tool is needed to build the extra evaluation file. `select_heldout` scoring is reused; `derive_mixture` becomes a fallback.

## Revisit when

The owner chooses the sources, or a design of the leave-tasks-out kind becomes necessary (for example if too few extra tasks exist).

## Related

* [D25](/design/d25-leave-tasks-out-for-unseen-label-sets.md)
* [D02](/design/d02-permissive-licences-only.md)
* [D13](/design/d13-bekko-licence-and-test-handling.md)
