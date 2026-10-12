---
type: Decision
title: 'D27: The data root is configurable (MIMIR_DECIDE_DATA)'
description: Data, mixtures and caches live in the folder named by the environment variable MIMIR_DECIDE_DATA, with ~/mimir-decide-data only as the fallback. On the server this is a persistent folder under /work.
status: accepted
date: '2026-10-12'
decided_by: agent (after the owner found the server layout)
tags: [process, environment, paths]
timestamp: 2026-10-12T00:00:00Z
---

## Context

All commands and configs assumed the data root `~/mimir-decide-data`. On the GPU server the home folder (`/home/ucloud`) is **not persistent**; the pilot mixture is in `/work/dfm/lukasgp/mimir-decide/data/mixture-v0`. A command with `DATA=$HOME/mimir-decide-data/mixture-v0` therefore pointed to a folder that does not hold the data, `build_extra_eval` stopped, and the audit failed on a missing manifest.

## Decision

- One setting: the environment variable `MIMIR_DECIDE_DATA`. When it is not set, the data root is `~/mimir-decide-data` (as before).
- Configs refer to the root with the placeholder `{DATA_ROOT}` (`configs/data.yaml`, `configs/train_*.yaml`); `mimir_decide.paths.load_config` replaces it. The MASSIVE cache default and `scripts/run_pilot.sh` follow the same variable.
- The experiment pages use `DATA=${MIMIR_DECIDE_DATA:-$HOME/mimir-decide-data}/mixture-v0`, so the same commands work on both machines.
- Server settings (the paths are the owner's): keep them in a small file that git ignores and that lives in the persistent repository folder, for example `/work/dfm/lukasgp/mimir-decide/.env.sh`, and run `source .env.sh` at the start of each session (a `~/.bashrc` in the non-persistent home would be lost):

```bash
export MIMIR_DECIDE_DATA=/work/dfm/lukasgp/mimir-decide/data
export HF_HOME=/work/dfm/lukasgp/hf-cache     # keeps the Hugging Face downloads (3.6 GB model, datasets) across sessions
```

## Consequences

- Nothing in the repository names a server path. `.env.sh` is in `.gitignore`.
- Runs still go to `runs/` in the repository ([D24](/design/d24-run-directories-in-the-repository.md)), which is persistent on the server. The pilot checkpoints must also be in a persistent place; check that none are only under `/home/ucloud`.
- `--set data_dir=...` and `--output_dir` still override the placeholder.

## Revisit when

A third machine needs a different layout, or the Hugging Face cache needs its own variable.

## Related

* [D20](/design/d20-environment-outside-onedrive.md)
* [D24](/design/d24-run-directories-in-the-repository.md)
