---
type: Decision
title: 'D23: Use the active Python environment'
description: Documents and scripts call python from the active environment, a uv .venv in the project root or an activated conda environment. No fixed venv path.
status: accepted
date: '2026-10-06'
decided_by: user
tags: [process, environment]
timestamp: 2026-10-06T00:00:00Z
---

## Context

The first setup used a fixed environment path (`UV_PROJECT_ENVIRONMENT=$HOME/.venvs/mimir-decide`) because the project folder on the Mac is synced by OneDrive. On the server this is not needed: there is either a uv environment in the project root or an activated conda environment. The owner asked to document this instead.

## Decision

- Commands in `README.md`, `AGENTS.md`, the experiment pages and `scripts/run_pilot.sh` use `python` from the active environment. Run them from the project root.
- Option A (uv): `uv sync --extra dev` creates `.venv` in the project root. Activate it (`source .venv/bin/activate`) or put `uv run` before a command.
- Option B (conda): activate the environment, then run `pip install -e ".[dev]"`.
- `UV_PROJECT_ENVIRONMENT` is not used in the documents any more. `scripts/run_pilot.sh` reads `PY` (default `python`) when a different interpreter is needed.
- Do not create a second environment if one exists.

## Consequences

- The right environment must be active before a command runs. A wrong interpreter fails at import, not silently.
- `.venv/` is in `.gitignore`. On a machine whose project folder is synced (the OneDrive Mac), a `.venv` in the project root would sync. Set `UV_PROJECT_ENVIRONMENT` yourself there. This is a local choice of the owner and not part of the documented workflow.
- The conda route was checked only by a dry-run resolution of `pip install -e ".[dev]"` (through `uv pip`). It was not installed into a real conda environment.
- Data, caches and runs still default to `~/mimir-decide-data` ([D20](/design/d20-environment-outside-onedrive.md)).

## Revisit when

The cluster needs a module system, containers or a CUDA build of torch that `uv.lock` does not provide.

## Related

* [D20](/design/d20-environment-outside-onedrive.md)
* [D04](/design/d04-single-gpu-plain-loop.md)
