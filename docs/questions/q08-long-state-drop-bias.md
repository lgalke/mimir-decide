---
type: Question
title: 'Q08: Does dropping long states bias the mixture?'
description: States over 12,000 characters are dropped; about a quarter of sampled bekko rows were.
status: open
priority: low
resolved_by: [e01]
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Why it matters

Retrieval, long-document and agent-trace tasks are underrepresented ([D17](/design/d17-sequence-and-option-budget.md)).

## What we know

1,037 bekko rows exceeded the limit in the trial build ([O02](/observations/o02-bekko-long-states.md)).

## How to resolve

Count dropped rows per source in the full build; raise `max_state_chars` and `max_len` if memory allows.

## Decision impact

Task coverage and what 'decision model' means for long inputs.
