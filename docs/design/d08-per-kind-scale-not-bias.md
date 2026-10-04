---
type: Decision
title: 'D08: Per-kind logit scale, no per-kind bias'
description: A learned scale per kind (noul, choice, score); a bias would be invisible to the softmax.
status: accepted
date: '2026-10-04'
decided_by: agent
tags: [model]
timestamp: '2026-10-04T00:00:00Z'
---

## Context

The first plan said 'per-kind bias/scale'. A constant added to all slots of one decision does not change a softmax.

## Decision

`log_scale` per kind multiplies the logits in training; a post-hoc temperature per kind is fitted afterwards ([D19](/design/d19-calibration-protocol.md)).

## Consequences

Scale and temperature overlap in effect; the temperature is the quantity that is reported.

## Revisit when

Never needed.

## Related

* [heads-and-baseline](/design/heads-and-baseline.md)
