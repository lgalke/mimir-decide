---
type: Question
title: 'Q10: Does HRM recurrence help the decision readout?'
description: Unknown whether more H/L cycles improve decisions or only cost compute.
status: open
priority: medium
resolved_by: [e09, e06]
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Why it matters

The colleague noted recurrence may help but does not remove backbone cost.

## What we know

The model runs 2 H and 6 L stack executions per pass ([HRM-Text](/research/hrm-text.md)).

## How to resolve

[E09](/experiments/e09-recurrence-depth-probe.md) (exploratory; needs a small code change).

## Decision impact

Whether an early-exit / adaptive-depth design is worthwhile.
