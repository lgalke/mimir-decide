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

**First evidence (2026-10-10):** for the zero-shot model, temperatures fitted on the calibration half cut the validation ECE (0.069 to 0.032) but worsened the held-out nb-NO result (ECE 0.029 to 0.047, NLL 0.299 to 0.310). One case, not yet checked for the fine-tuned runs, which have no raw held-out evaluation ([O16](/observations/o16-zero-shot-baseline-fine-tuning-adds-16-points.md)).

## How to resolve

[E04](/experiments/e04-calibration-effect.md): ECE on heldout vs validation, before and after.

## Decision impact

Whether a confidence threshold set on validation is safe elsewhere.
