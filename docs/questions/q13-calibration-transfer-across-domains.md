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

**Fine-tuned runs (2026-10-11):** temperatures do not help on the held-out task: slot ECE 0.021 raw to 0.027 calibrated, baseline 0.018 to 0.019 (NLL improves by 0.002 and 0.001). The raw ECE is already small, so there is little to transfer ([O18](/observations/o18-temperature-scaling-barely-changes-the-fine-tuned-models.md)). Still one held-out task, which Mimir probably saw.

## How to resolve

[E04](/experiments/e04-calibration-effect.md): ECE on heldout vs validation, before and after.

## Decision impact

Whether a confidence threshold set on validation is safe elsewhere.
