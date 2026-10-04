---
type: Decision
title: 'D15: Held-out task: massive/nb-NO (provisional)'
description: One MASSIVE locale is removed from training and evaluated separately.
status: accepted
date: '2026-10-04'
decided_by: agent
tags: [evaluation, provisional]
timestamp: '2026-10-04T00:00:00Z'
---

## Context

The colleague advised holding out whole tasks and domains, not random rows.

## Decision

`held_out_sources: ["massive/nb-NO"]` in `configs/data.yaml`; its validation and test rows go to `heldout_tasks/heldout.parquet`.

## Consequences

**Caveat found afterwards:** tasksource contains `multilingual/massive` and Mimir trained on `tasksource__`, so nb-NO may have been seen ([O08](/observations/o08-audit-flag-rates.md)). It still tests transfer to an unseen *locale of our conversion*, not unseen data. Only LocalLLaMA is unflagged.

## Revisit when

Before the first real run: pick held-out tasks that are not flagged, or accept the weaker meaning ([Q01](/questions/q01-clean-generalisation-evidence.md)). Change it only before any training and log the change.

## Related

* [dataset-selection](/design/dataset-selection.md)
