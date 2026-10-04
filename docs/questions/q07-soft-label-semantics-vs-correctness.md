---
type: Question
title: 'Q07: Are soft labels probabilities of being correct?'
description: Human disagreement and teacher probabilities are not objective correctness probabilities.
status: open
priority: medium
resolved_by: [e04]
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Why it matters

Calibration is measured against the labels we have ([D19](/design/d19-calibration-protocol.md)).

## What we know

The colleague flagged this. LocalLLaMA has teacher-style soft labels; most tasksource targets are hard one-hot.

## How to resolve

Report calibration separately on hard-label and soft-label sources ([E04](/experiments/e04-calibration-effect.md)); do not call the output 'probability of correctness' without a ground-truth check.

## Decision impact

Wording of any calibration claim.
