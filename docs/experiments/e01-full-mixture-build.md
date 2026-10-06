---
type: Experiment
title: 'E01: Full mixture build'
description: Build the real mixture and record what survives licences, leakage guards and caps.
status: planned
answers: [q06, q08]
depends_on: []
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Hypothesis

The strict licence policy and guards still leave roughly 500k-1M train decisions across hundreds of sources.

## Prerequisites

none (network, disk, hours)

## Setup

```bash
# Run from the repository root with the project environment active (uv .venv or conda).
export DATA=$HOME/mimir-decide-data/mixture-v0 RUNS=$HOME/mimir-decide-data/runs
PY=python
```
```bash
$PY -m mimir_decide.build_mixture --config configs/data.yaml --output_dir $DATA 2>&1 | tee $DATA.build.log
$PY -m mimir_decide.audit_mimir_overlap --data_dir $DATA
```
Use `--limit N` for a smoke build first. Reads tasksource by streaming (all splits) and bekko parquet files via the HF cache.

## Metrics to record

From `mixture_manifest.json`: counts per split; `train_by_kind`, `train_by_dataset`, `train_sources`; `drops` per collection (licence, ngram, duplicate, state_too_long); `heldout_patterns_unmatched` (must be empty); `train_sources_without_eval_rows`; `share_alike_train_sources`; `reports/ngram_leak_examples.jsonl` (inspect 20 by hand); `reports/license_exclusions.csv`; `reports/mimir_overlap.md`.

## Decision rule

Proceed to E02 if train >= 300k decisions, `heldout_patterns_unmatched` is empty and the manual n-gram inspection shows real duplicates. If train is far below 300k, discuss admitting bekko 'qualified' subsets or more tasksource sources with the owner (Q06) before training.

## Status and results

Status: **planned**. Results: _not run yet_.

| Date | Run id / path | Config (overrides) | Result | Observation page |
|---|---|---|---|---|
| | | | | |

When run: fill this table, create an observation page (copy the template) with the numbers, set `status` in the frontmatter, add a line to the update log, and regenerate the indexes.
