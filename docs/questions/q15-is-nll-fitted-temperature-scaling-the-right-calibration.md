---
type: Question
title: 'Q15: Is NLL-fitted temperature scaling the right calibration?'
description: The fitted temperatures lower NLL slightly but raise ECE on noul and held-out data. Another fitting criterion, or no post-hoc step for the fine-tuned models, may be better.
status: open
priority: low
resolved_by: [e04]
tags: []
timestamp: 2026-10-11T00:00:00Z
---

## Why it matters

[D19](/design/d19-calibration-protocol.md) fits one temperature per kind by minimizing NLL on the calibration half. For the fine-tuned models this step brings almost nothing and makes noul ECE worse ([O18](/observations/o18-temperature-scaling-barely-changes-the-fine-tuned-models.md)).

## What we know

Raw ECE of the fine-tuned models is 0.015 to 0.019. The fitted noul temperatures (1.23 to 1.27) lower the confidence of a model that is already slightly underconfident. The best NLL temperature is not the best ECE temperature when a few confident errors dominate the loss.

## How to resolve

Compare three options on validation and held-out data, with the same metrics: no post-hoc step for fine-tuned models, a temperature fitted on ECE (or on a binned objective), and a more flexible method fitted on the calibration half. Needs only evaluation runs and a small change in `calibrate.py`. Do not pick the method on the test split.

## Decision impact

Whether D19 is amended: for example to skip the post-hoc step when the raw ECE is already below a threshold.
