---
type: Observation
title: 'O27: Mixture v1 (1M) is 2.7 times larger but has the same 366 sources; two composition targets were missed'
description: The 1M build keeps the evaluation identical to v0 and reaches noul 19.6%, but score stays at 11.6% and the binary share rises to 43.2%. The licence fix admitted no rows at the pinned tasksource revision.
date: '2026-10-22'
confidence: high
tags: [e16, mixture, v1, diversity, build]
timestamp: 2026-10-22T00:00:00Z
---

## Observation

Source: the build of `configs/data_v1.yaml` on the server with `--set sampling.total=1000000 per_source_cap=12000`, output pasted by the owner (`mixture_report --compare mixture-v0` and manifest keys). Only these keys were pasted; the full manifest and the build log were not.

| | v1 (1M) | v0 |
|---|---|---|
| Train decisions | 999,906 | 371,275 |
| Sources | 366 | 366 |
| Effective sources | 250.4 | 255.6 |
| Largest source / ten largest | 0.62% / 6.2% | 0.54% / 5.4% |
| Sources with at least 1,000 decisions | 225 | 171 |
| Kind: choice / noul / score | 68.8% / 19.6% / 11.6% | 73.1% / 16.7% / 10.2% |
| Binary decisions (two options) | **43.2%** | 41.8% |
| Soft targets; mean top label mass | 6.1%; 0.982 | 6.4%; 0.981 |
| Characters p50 / p90 / p99 of state and options | 289 / 1,877 / 5,932 | 322 / 2,097 / 6,423 |
| Script mix (dominant script of the state) | Latin 97.8%, Hangul 0.6%, Arabic 0.4%, Cyrillic 0.3% | Latin 98.7% |

By dataset (v1): tasksource 778,551 (69.9%), bekko 162,956 (16.3%), procedural generators 8.0% (inside tasksource), HelpSteer2 30,980, MASSIVE 24,015, LocalLLaMA 3,404 (0.34%; the same as v0, all that survives the guards). Splits: validation 6,127, calib 6,189, heldout 600, test 11,674, all identical to v0 (`frozen_eval_identical_to_reference`: True for the four files). `heldout_patterns_unmatched`: empty. `newly_admitted_by_annotation_stripping`: **0**. `sampling.converged`: **False**.

## Interpretation

- **Size and evaluation as planned:** 2.7 times more decisions, evaluation files identical, held-out pattern matched, no leakage assertion failed.
- **No gain in source diversity.** The 366 sources are the same count as in v0; the rows per source rose (225 sources with at least 1,000 decisions against 171), and the largest-source share is unchanged. Source diversity is bounded by the licence policy: the 262 tasksource sources with unknown licences (822,420 rows in the current `full`) are excluded ([D02](/design/d02-permissive-licences-only.md)).
- **The licence fix did nothing here.** My estimate of +89,995 rows came from the `sources.yaml` of the current dataset head, where licences carry annotations such as `(original source)`. The build pins the older revision `d2ab1d12` (2026-10-02), whose rows carry plain licence names. At that revision the stripping admits zero rows ([D32](/design/d32-licence-annotation-stripping.md) updated). Using the newer revision would need a separate decision, because the evaluation files must stay identical.
- **Composition targets partly missed.** Noul 19.6% (floor 20%, missed by 0.4 points); score 11.6% against the floor of 15%: the pool has too few score decisions in permissive sources (score questions come mostly from HelpSteer2, which is capped by its size, and from few tasksource sources); binary decisions 43.2% against a cap of 40%, and higher than v0. The binary penalty applies to binary choice and score decisions, but noul decisions (all binary) are boosted to reach their floor, so the two targets pull against each other.
- **Not yet checked:** the diversity gain in task types (the report counts sources, not tasks), and whether the 163k bekko decisions overlap in task with tasksource sources (cross-source dedup removes only exact and n-gram duplicates).

## Related

* [E16](/experiments/e16-mixture-v1-training.md)
* [D31](/design/d31-mixture-v1-size-and-diversity.md)
* [O25](/observations/o25-full-mixture-build-counts-and-drops.md)
