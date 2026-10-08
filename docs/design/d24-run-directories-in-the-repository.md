---
type: Decision
title: 'D24: Run directories live in the repository'
description: Runs default to runs/ in the repository, so the tracked result files (eval JSON, training log) are written where git sees them. Checkpoints stay ignored by git.
status: accepted
date: '2026-10-09'
decided_by: user
tags: [process, runs, git]
timestamp: 2026-10-09T00:00:00Z
---

## Context

[D20](/design/d20-environment-outside-onedrive.md) put runs under `~/mimir-decide-data/runs`, outside the repository. The owner then decided that the small result files are tracked in git ([D22](/design/d22-git-workflow.md)): `runs/<run>/eval/*.json` and `runs/<run>/train_log.jsonl`. With runs outside the repository, someone had to copy these files by hand. The owner asked to change the default run location.

## Decision

- Runs default to `runs/` in the repository: `run_dir: runs/slot-v0` and `runs/baseline-v0` in the training configs, and `RUNS` defaults to `$PWD/runs` in `scripts/run_pilot.sh`. The paths are relative to the repository root, so commands must run from there.
- The experiment pages use `RUNS=runs`.
- `.gitignore` is unchanged: it tracks only `runs/*/eval/*.json` and `runs/*/train_log.jsonl` and ignores everything else under `runs/`.
- Data and caches stay in `~/mimir-decide-data` (`DATA`, `cache_dir`).
- The baseline config now names its run `baseline-v0` (it was `letter-v0`), like the pilot script.

## Consequences

- Results and training curves appear in `git status` right after a run. Check their size before committing, as `AGENTS.md` section 9 says.
- **Checkpoints now use disk space inside the repository folder:** `best/` and `final/` are about 3.6 GB each, so about 7 GB per run, ignored by git. Check the free space where the repository is, or set `RUNS` (or `--set run_dir=...`) to another place. On a machine where the repository folder is synced (the OneDrive Mac), always set `RUNS` elsewhere.
- The pilot checkpoints were written to the old location on the server. To use them with the new default, copy them without overwriting the tracked files, for example `rsync -a --ignore-existing ~/mimir-decide-data/runs/slot-v0/ runs/slot-v0/`. This copy was not tested. Experiments that need the pilot checkpoints (E04, E05, E08) need it, or `RUNS=$HOME/mimir-decide-data/runs`. E10's comparison needs only the tracked eval files.
- Old documents that mention the old path were updated; the remaining mentions (D20 note, `AGENTS.md` migration hint) are intended.

## Revisit when

Disk space in the repository folder is too small for the checkpoints, or the owner wants checkpoints kept apart again.

## Related

* [D20](/design/d20-environment-outside-onedrive.md)
* [D22](/design/d22-git-workflow.md)
