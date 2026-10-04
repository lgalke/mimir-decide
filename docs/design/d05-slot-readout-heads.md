---
type: Decision
title: 'D05: One shared slot readout for Noul, Choice and Score'
description: Options are marked slots in the prompt; one scalar logit per option read from the final H-state; no per-task heads.
status: accepted
date: '2026-10-04'
decided_by: user (accepted proposal)
tags: [model, heads]
timestamp: '2026-10-04T00:00:00Z'
---

## Context

Jev takes questions and options at inference time, so fixed per-task heads cannot reproduce it. The owner asked whether three separate heads are a good idea; the proposal was to share one open-vocabulary readout.

## Decision

See [heads and baseline](/design/heads-and-baseline.md): Choice = softmax over slots; Noul = two slots; Score = ordered bins with a squared-CDF penalty; per-kind scale and temperature; independent sequences per decision; option order shuffled in training.

## Consequences

- Option sets are open vocabulary; the same weights handle any question.
- Cost is one forward pass per decision ([Q12](/questions/q12-option-position-bias-at-evaluation.md) for position bias).

## Revisit when

[E03](/experiments/e03-slot-vs-letter-pilot.md) shows no advantage over the baseline.

## Related

* [heads-and-baseline](/design/heads-and-baseline.md)
* [decision-models](/research/decision-models.md)
