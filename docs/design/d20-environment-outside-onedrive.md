---
type: Decision
title: 'D20: Environment and data live outside OneDrive'
description: Venv in ~/.venvs/mimir-decide; data, caches, runs in ~/mimir-decide-data.
status: accepted
date: '2026-10-04'
decided_by: agent
tags: [process, environment]
timestamp: '2026-10-04T00:00:00Z'
---

## Context

The project folder is synced by OneDrive; a venv or checkpoints there would sync thousands of files and gigabytes.

## Decision

`UV_PROJECT_ENVIRONMENT=$HOME/.venvs/mimir-decide`; data root `~/mimir-decide-data` (override with `DATA`, `RUNS`, `output_dir`, `cache_dir`).

## Consequences

On the GPU cluster, set these to fast local storage; paths in `configs/*.yaml` use `~` and can be overridden with `--set`.

## Revisit when

Never.

## Related

* [training-setup](/design/training-setup.md)
