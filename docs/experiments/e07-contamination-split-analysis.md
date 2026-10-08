---
type: Experiment
title: 'E07: Seen vs not-flagged analysis'
description: Separate every result by whether Mimir's mixture contains the source (name-level).
status: planned
answers: [q01]
depends_on: [e03]
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Hypothesis

Performance on sources flagged as seen by Mimir is higher than on unflagged ones, i.e. part of the apparent skill is familiarity.

## Prerequisites

E03 runs and `reports/mimir_overlap.json`

## Setup

```bash
# Run from the repository root with the project environment active (uv .venv or conda).
export DATA=$HOME/mimir-decide-data/mixture-v0 RUNS=runs
PY=python
```
```bash
$PY -m mimir_decide.compare $RUNS/slot-v0 $RUNS/baseline-v0 --file validation.json
```
The `[seen_by_mimir]` and `[not_flagged]` blocks pool sources by n. Only LocalLLaMA is unflagged in the trial audit, so also compare per source.

## Metrics to record

Pooled and per-source accuracy and NLL for seen vs not flagged; sample sizes.

## Decision rule

Any generalisation statement must cite the not-flagged numbers and their n. If there is no sizeable unflagged set, say so and pursue Q01 options.

## Status and results

Status: **planned**. Results: _not run yet_.

| Date | Run id / path | Config (overrides) | Result | Observation page |
|---|---|---|---|---|
| | | | | |

When run: fill this table, create an observation page (copy the template) with the numbers, set `status` in the frontmatter, add a line to the update log, and regenerate the indexes.
