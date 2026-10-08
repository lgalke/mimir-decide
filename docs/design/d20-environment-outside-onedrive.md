---
type: Decision
title: 'D20: Environment and data live outside OneDrive'
description: Data, caches and runs live in ~/mimir-decide-data, outside the synced folder. The venv-location part is superseded by D23.
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

## Update 2026-10-06

The venv location (`~/.venvs/mimir-decide`, `UV_PROJECT_ENVIRONMENT`) is superseded by [D23](/design/d23-active-python-environment.md): documents and scripts now use the active Python environment. The data and cache locations in this decision are unchanged. The run location (`~/mimir-decide-data/runs`) is superseded by [D24](/design/d24-run-directories-in-the-repository.md).

## Revisit when

Never.

## Related

* [training-setup](/design/training-setup.md)
