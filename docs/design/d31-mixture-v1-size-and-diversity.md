---
type: Decision
title: 'D31: Mixture v1: larger and more diverse training set, frozen evaluation, 4096 context'
description: configs/data_v1.yaml builds a pool from tasksource full and a wider bekko read, with per-source, family, kind and binary-share targets and nested named sizes. The evaluation files stay identical to mixture-v0.
status: accepted
date: '2026-10-21'
decided_by: owner (goal, context 4096, two-step sizing); design by agent
tags: [data, mixture, diversity, v1]
timestamp: 2026-10-21T00:00:00Z
---

## Context

`mixture-v0` has 371,275 train decisions from 366 sources, set by caps and not by the available data: `per_source_cap: 2000` and `read_limit_per_source: 20000` ([O25](/observations/o25-full-mixture-build-counts-and-drops.md)). The builder read only the `default` config of tasksource (999,832 train rows). The same dataset has `full` (2,550,012 train rows; 1,235,637 of them permissive by our policy), and bekko holds roughly 2M permissive train rows that v0 never read (1,914,885 dropped by the read limit). The owner asked for a larger and more diverse mixture, plans a 4096 context, and plans faster training on a B200.

## Decision

- **Evaluation is frozen.** `validation`, `calib`, `heldout`, `eval/test` of `mixture-v1` must equal those of `mixture-v0`; the build checks content digests (`frozen_eval_from`) and fails otherwise. The eval pass keeps v0's filters (`eval_max_state_chars: 12000`, the strict licence parser, tasksource `default` for validation/test), so old and new runs stay comparable, as do the Typed Decisions test and the E12 extra set.
- **Guards are not weakened ([D12](/design/d12-ngram-fingerprint-guard-cross-source-only.md)).** One addition: eval states removed only by the length filter, and the validation/test rows of tasksource `full`, are also hashed for the guards (`guard_long_eval_states`, `guard_eval_configs`), because train rows can now be longer.
- **Exclusions:** `massive/nb-NO` (held out) and all sources of `configs/extra_eval_sources.yaml` (E12) never enter train.
- **Pool:** tasksource `full` at the pinned revision, bekko with `read_limit_per_source: 60000`, MASSIVE, HelpSteer2 and LocalLLaMA as before. Train rows up to `train_max_state_chars: 16000` (about 4000 tokens); the formatter's middle cut still applies above the budget.
- **Licences ([D32](/design/d32-licence-annotation-stripping.md)):** `license_strip_annotations: true` in the train pass only.
- **Sampling (`sampling` section, code in `mimir_decide/sampling.py`):** quotas `min(n_s, c n_s^0.5, 1% of N)` solved by bisection; family caps (procedural generators 10%, bekko 25%); weights that raise noul to at least 20% and score to at least 15% (v0: 16.7% and 10.2%) and keep binary decisions (two options) at most 40%, found by simulation with the real selection rule. A floor that the data cannot supply is reported (`converged: false`, a warning), not hidden.
- **Nested sizes:** the full set (target 2,000,000, fewer if the pool is smaller) and named subsets `v1-371k` (v0's size) and `v1-1M`, written as `train_<name>.parquet`. Every subset is contained in every larger one, by a fixed priority order per source. Training picks one with `train_file` in the config.
- **Context:** training configs `train_slot_v1.yaml` and `train_baseline_v1.yaml` use `max_len: 4096` and the `v1-371k` subset.

## Consequences

- The first training run on v1 (E16) compares composition at equal size (v1-371k against v0) but also changes the context length; the E16 page names this confound and the control.
- Building the pool needs more memory: up to about 3M decision objects in RAM in the reservoirs (estimate, not measured), then the parquet writer works in chunks.
- The binary cap and the floors are targets: the report shows what was reached for every size. Subsets use the boosts solved on the full set, so their composition can drift a little; a warning is printed below 3 points under a floor.
- Diversity is measured by `python -m mimir_decide.mixture_report` (sources, effective sources, top shares, kind and option-count mix, target entropy, script mix).
- Not run on the real data yet. Unit tests cover quotas, caps, nesting, floors, frozen evaluation, exclusions, the long-state guard and licence stripping; a small trial build was run (see the log).

## Revisit when

The v1 build shows a pool smaller than expected, a floor that cannot be met, or an effect of the families on E12; or the B200 throughput changes the size that is worth training.

## Related

* [D12](/design/d12-ngram-fingerprint-guard-cross-source-only.md)
* [D17](/design/d17-sequence-and-option-budget.md)
* [D02](/design/d02-permissive-licences-only.md)
* [E16](/experiments/e16-mixture-v1-training.md)
