---
type: Decision
title: 'D06: Letter-logit baseline'
description: Compare against reading A/B/C letter logits from the unchanged language-model head.
status: accepted
date: '2026-10-04'
decided_by: user
tags: [model, baseline]
timestamp: '2026-10-04T00:00:00Z'
---

## Context

The owner asked for the baseline alongside the heads strategy.

## Decision

`LetterBaseline`: Gemma chat prompt with lettered options and an instruction to answer with the letter; logits of the letter tokens at the last position; same data, loss, calibration and metrics. LM head and embeddings frozen by default.

## Consequences

- Options are capped at 26 ([D17](/design/d17-sequence-and-option-budget.md)).
- The prompt was verified identical to `apply_chat_template` by a network test ([O11](/observations/o11-letter-prompt-tokenization-boundary.md)).

## Revisit when

The baseline is unfairly handicapped (for example frozen head) and the comparison needs a stronger variant (`train_lm_head: true`).

## Related

* [heads-and-baseline](/design/heads-and-baseline.md)
