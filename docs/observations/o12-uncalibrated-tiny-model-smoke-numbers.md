---
type: Observation
title: 'O12: Smoke-run metrics on the random tiny model are meaningless'
description: Do not read the tiny-model accuracy, NLL or temperatures as results.
date: '2026-10-04'
confidence: high
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Observation

Slot and letter runs on a random 2-layer model gave accuracy ~0.30-0.33, NLL ~1.32 and degenerate temperatures (some at the 0.05 or 20 clamp). They only prove that build, train, save, load, calibrate and evaluate run.

## Interpretation

Any number from the real model must come from E02 onwards.

## Reproduce

`scripts/make_tiny_model.py` and the smoke commands in the log.

## Related

* [training-setup](/design/training-setup.md)
