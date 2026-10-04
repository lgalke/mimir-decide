---
type: Observation
title: 'O02: About a quarter of sampled bekko rows exceed 12,000 characters'
description: Retrieval-style bekko tasks are mostly dropped by the length limit.
date: '2026-10-04'
confidence: medium
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Observation

Trial build: `state_too_long` dropped 1,037 bekko decisions (of roughly 4,000 read); tasksource lost only 21.

## Interpretation

Long-document and retrieval tasks are underrepresented in the training mixture.

## Reproduce

Same trial build; `drops.bekko.state_too_long` in the manifest.

## Related

* [d17-sequence-and-option-budget](/design/d17-sequence-and-option-budget.md)
* [q08-long-state-drop-bias](/questions/q08-long-state-drop-bias.md)
