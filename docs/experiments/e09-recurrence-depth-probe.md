---
type: Experiment
title: 'E09: Recurrence depth probe (exploratory)'
description: Evaluate the trained model with fewer H cycles to see whether depth matters for decisions.
status: planned
answers: [q10]
depends_on: [e03]
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Hypothesis

Reducing `H_cycles` at inference lowers decision quality only slightly, so adaptive depth could save compute.

## Prerequisites

E03 run; needs a small code change (not implemented)

## Setup

Not implemented. Needs a flag in `evaluate.py` / `load_model` that overrides `config.H_cycles` (and `L_cycles`) at load time, plus a check that the recurrence code is valid with changed cycle counts at inference (the cache-slot layout depends on them; use `use_cache=False`). Verify on the tiny model first.

## Metrics to record

Accuracy and NLL vs `H_cycles` in {1, 2}; wall-clock per decision.

## Decision rule

Only pursue if E03 shows the slot model is worth keeping. Record the code change as a decision.

## Status and results

Status: **planned**. Results: _not run yet_.

| Date | Run id / path | Config (overrides) | Result | Observation page |
|---|---|---|---|---|
| | | | | |

When run: fill this table, create an observation page (copy the template) with the numbers, set `status` in the frontmatter, add a line to the update log, and regenerate the indexes.
