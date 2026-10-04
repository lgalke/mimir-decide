---
type: Decision
title: 'D14: Carving validation/test where upstream has none'
description: Deterministic group-wise carve-outs by hashed id.
status: accepted
date: '2026-10-04'
decided_by: agent
tags: [data, splits]
timestamp: '2026-10-04T00:00:00Z'
---

## Context

LocalLLaMA has only train/test; HelpSteer2 only train/validation; MASSIVE calls validation 'dev'.

## Decision

LocalLLaMA: 10% of upstream train (by row id) becomes validation; upstream test stays test. HelpSteer2: 5% of upstream train (by prompt) becomes validation; upstream validation becomes **test**. MASSIVE: dev is validation.

## Consequences

HelpSteer2 numbers on our test are comparable with the standard reward-model benchmark. Carving is by hashed key so it is reproducible.

## Revisit when

Never, unless the datasets change.

## Related

* [dataset-selection](/design/dataset-selection.md)
