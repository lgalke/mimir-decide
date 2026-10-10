---
type: Question
title: 'Q14: How much of the fine-tuned baseline''s quality comes from fine-tuning?'
description: The letter baseline reaches about 0.835 accuracy after fine-tuning. How much would the untrained Mimir already reach with the same prompt?
status: answered
priority: high
resolved_by: [e10]
tags: []
timestamp: 2026-10-09T00:00:00Z
---

## Why it matters

Without a zero-shot reference, the strength of the baseline cannot be explained. If the untrained model is already close, the pilot compares two readouts on a model that barely changed. If it is far below, fine-tuning is the main source of quality.

## What we know

The fine-tuned baseline and the slot model are tied ([O14](/observations/o14-first-pilot-results-slot-and-baseline-tie.md)). 96% of validation is from sources in Mimir's training mix ([O08](/observations/o08-audit-flag-rates.md)). There is no result for the untrained model.

## How to resolve

[E10](/experiments/e10-zero-shot-letter-baseline.md): evaluate Mimir v1.5 with the letter prompt and no training, raw and with 3 fitted temperatures.

## Answer

Most of it. The untrained Mimir v1.5 with the letter prompt reaches 0.669 validation accuracy; the fine-tuned baseline reaches 0.835, a gain of 16.6 points (Score +30, noul +15, choice +14). On the held-out MASSIVE task the gain is about 6 points (0.895 to 0.952). Part of the gain is learning task formats and label conventions, and the zero-shot number is a lower bound because the prompt is ours ([O16](/observations/o16-zero-shot-baseline-fine-tuning-adds-16-points.md)).

## Decision impact

Wording of the conclusions in O14 and the interpretation of E03 and E11.
