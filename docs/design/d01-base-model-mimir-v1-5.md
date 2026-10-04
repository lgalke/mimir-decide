---
type: Decision
title: 'D01: Base model is Mimir v1.5'
description: Start from DFM Mimir v1.5 (pinned revision), not v1.
status: accepted
date: '2026-10-04'
decided_by: user
tags: [model]
timestamp: '2026-10-04T00:00:00Z'
---

## Context

The first plan used Mimir v1 (1B). On 2026-10-04 the project owner asked to use the v1.5 snapshot instead.

## Decision

Base = `danish-foundation-models/DFM-Mimir-v1.5` at revision `521b40b36a79918014544b970d4c2669ff1530eb` (config `base_model`, `revision`). Tokenizer from the same revision.

## Consequences

- 1.787B parameters in total (0.98B outside the untied 262k embedding and output head); more memory than v1.
- BF16 inference weights only: no optimizer state, so this is a fresh fine-tune, not a resumed run.
- `L_bp_cycles` is `[3,3]` (see [D09](/design/d09-keep-checkpoint-l-bp-cycles.md)).
- The repo ships its training-data manifest and sampling policy, which enabled the overlap audit ([O07](/observations/o07-mimir-policy-includes-tasksource-and-flan.md)).

## Revisit when

A pre-instruction-tuning base checkpoint appears, or v1 is needed as a comparison.

## Related

* [mimir-v1-5](/research/mimir-v1-5.md)
* [mimir-v1](/research/mimir-v1.md)
