---
type: Decision
title: 'D03: Skip in-house data for now'
description: DaLA, GEC, Arena, audit and tool examples are not converted yet.
status: accepted
date: '2026-10-04'
decided_by: user
tags: [data, scope]
timestamp: '2026-10-04T00:00:00Z'
---

## Context

The Mimir training lead suggested in-house data as a relevant multilingual specialisation ([colleague recommendations](/research/colleague-recommendations.md)). The owner chose to skip it for the pilot.

## Decision

No converters for DaLA acceptability, GEC pairs, Arena preferences, audits (keep/repair/reject/needs_verification) or tool-choice examples. The pilot uses public sources only.

## Consequences

- No data that Mimir demonstrably has not seen, so no clean generalisation test ([Q01](/questions/q01-clean-generalisation-evidence.md)). Note that `giannor_dala_*`, `giannor_gec_*` and Arena data are in Mimir's own policy.
- A converter interface exists (`convert/*.py`, `iter_records(split, drops, limit)` returning `Decision`) to add them later.

## Revisit when

The owner shares the data paths or formats.

## Related

* [colleague-recommendations](/research/colleague-recommendations.md)
