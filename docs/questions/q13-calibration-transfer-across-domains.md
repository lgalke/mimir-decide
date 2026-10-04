---
type: Question
title: 'Q13: Does calibration transfer to held-out sources?'
description: Laya's calibration was not shown to transfer across domains.
status: open
priority: medium
resolved_by: [e04]
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Why it matters

Temperatures are fitted on validation of the training sources ([D19](/design/d19-calibration-protocol.md)).

## What we know

Held-out tasks are evaluated with the same temperatures.

## How to resolve

[E04](/experiments/e04-calibration-effect.md): ECE on heldout vs validation, before and after.

## Decision impact

Whether a confidence threshold set on validation is safe elsewhere.
