---
type: Decision
title: 'D37: Crash-safe snapshots and resume for long training runs'
description: Option snapshot_every writes a replaceable resume/state.pt (parameters, optimizer, averaged weights, counters, data position); resume=true continues exactly from it. Added after the server stop of 2026-10-27 cost two long runs.
status: accepted
date: '2026-10-31'
decided_by: owner (resume support after the server stop); implementation by agent
tags: [training, resume, reliability]
timestamp: 2026-10-31T00:00:00Z
---

## Context

On 2026-10-27 the server stopped and killed two full runs at 84% and 67% of their steps (about 60 GPU hours). `train.py` had no resume, and `best/` and `final/` hold only bf16 inference weights without optimizer state.

## Decision

- **Config:** `snapshot_every: null` (off) and `resume: false` in the v1 configs. With `snapshot_every: N` the loop writes `run_dir/resume/state.pt` after every N optimizer steps (after the validation of that step), by writing a temporary file and renaming it, so a kill during the write leaves the previous snapshot intact. With `resume=true` the run continues from that file; if none exists it starts from the beginning and says so.
- **Content:** the trainable parameters (fp32), the optimizer state, the averaged weights (when `ema_decay` is set), the step, epoch and micro-batch counters, the position in the data (the number of batches already drawn in the epoch), elapsed time and the best validation values. The frozen weights are loaded from the base model again. The data order is fixed by `(seed, epoch)` and the per-item augmentation seeds, so skipping the consumed batches reproduces the run.
- **Guard:** the snapshot records the total number of steps and the number of parameters; a resume with a different configuration stops with an error.
- **Test and maintenance:** `stop_after_step: N` writes a snapshot at step N and exits cleanly.
- **Size and cost:** about 12 GB per snapshot for the 981M trainable parameters (parameters 3.9 GB, two Adam moments 7.8 GB; plus 3.9 GB with EMA), replaced each time, so one snapshot of disk per run; writing takes about a minute or two (not measured on the server). For 1000 steps of a 1M run (about 1.3 h) that is acceptable.

## Consequences

- A crash costs at most `snapshot_every` steps. After a crash, start the same command again with `resume=true` (same configuration).
- Tested on a tiny model: a run stopped at step 4 and resumed gives final raw and EMA weights identical to an uninterrupted run (maximum difference 0.0) and the same validation values. Not run on the real model; GPU non-determinism may make a resumed run differ slightly from an uninterrupted one.
- A resumed run appends to the same `train_log.jsonl`; steps after the last snapshot appear twice if the run died after logging them.
- `snapshot_every` is not set in the older pilot configs.

## Revisit when

The snapshot size or write time becomes a problem, or the run configuration changes between a crash and the resume (this needs a fresh run).

## Related

* [D36](/design/d36-optional-weight-average-ema.md)
* [training-setup](/design/training-setup.md)
