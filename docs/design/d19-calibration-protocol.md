---
type: Decision
title: 'D19: Calibration: one temperature per kind on held-out validation'
description: Fit temperatures by LBFGS on soft NLL using half of validation, split by group.
status: accepted
date: '2026-10-04'
decided_by: agent
tags: [calibration, evaluation]
timestamp: '2026-10-04T00:00:00Z'
---

## Context

Laya's ECE dropped from 0.466 to 0.081 with temperature scaling ([Laya](/research/laya.md)); a proper calibration set must differ from train and test.

## Decision

`calibrate.py` fits T per kind on `calib.parquet` (validation, disjoint by group from `validation.parquet`), clamped to [0.05, 20]; `evaluate.py` applies it. ECE uses 15 bins on top-1 confidence versus 'picked a maximal-target option'.

## Consequences

Calibration is to the labels given (possibly soft or teacher-made), not necessarily to truth. Temperatures at the clamp bounds indicate a degenerate fit and should be investigated.

## Revisit when

Calibration does not transfer to held-out sources ([Q13](/questions/q13-calibration-transfer-across-domains.md)).

## Related

* [calibration-training](/research/calibration-training.md)
