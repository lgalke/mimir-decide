---
type: Decision
title: 'D16: Exclude ChaosNLI'
description: Not used, not even for evaluation.
status: accepted
date: '2026-10-04'
decided_by: agent
tags: [data, licence]
timestamp: '2026-10-04T00:00:00Z'
---

## Context

The colleague suggested ChaosNLI (100 annotations per item) mainly for calibration research.

## Decision

Excluded: licence is CC-BY-NC, and it is built on SNLI/MNLI/alphaNLI dev sets, which are also eval material elsewhere.

## Consequences

No human-disagreement benchmark. LocalLLaMA and bekko carry soft labels, but these are not human distributions.

## Revisit when

Non-commercial research use is allowed, as an evaluation-only set with strict split control.

## Related

* [colleague-recommendations](/research/colleague-recommendations.md)
