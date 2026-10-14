---
type: Experiment
title: 'E15: Decision Index public suite'
description: Run our models through the public suite of the Decision Index (37 benchmarks, about 140k requests) with the kit's own scorer, after checking for overlap with our training mixture.
status: planned
answers: [q01]
depends_on: [e14]
tags: [benchmark, decision-index, external]
timestamp: 2026-10-14T00:00:00Z
---

## Hypothesis

None about a winner. The Index tests tasks far from our mixture (tool use, retrieval, knowledge, long context). Expectation, not a result: fine-tuned models gain over the untrained model on classification-like benchmarks that tasksource contains, and lose coverage on long prompts and on retrieval with more than 26 options (letter model).

## Prerequisites

The kit installed in the project environment (`pip install -e <checkout>` from [apolinario/decision-index](https://github.com/apolinario/decision-index), MIT licence); about 7 GB of downloads and 17 GB of working space; the HLE terms accepted on the Hub with a logged-in account; the adapter of [D29](/design/d29-decision-index-adapter.md); checkpoints of `slot-v0`, `baseline-v0`, `baseline-lmhead`, `letter-zeroshot` (and the probes) on the server.

## Setup

```bash
# Run from the kit checkout with the project environment active; the rebuild commands are those of the kit's README.
export HF_HUB_DISABLE_XET=1
python -m decision_index suite rebuild --work work
python -m decision_index suite import \
    --rows work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz \
    --added-rows work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz \
    --gsm8k-rows work/artifacts/benchmark-suite/release-v3-rebuilt/gsm8k-rows.jsonl.gz
python -m decision_index suite sample --n 100 --out sample-100.jsonl.gz       # smoke test first
python -m decision_index run --engine mimir_decide.decision_index_engine:MimirDecideEngine \
    --option run_dir=<repo>/runs/slot-v0 --option checkpoint=best \
    --rows sample-100.jsonl.gz --out <repo>/runs/di-smoke-slot-v0
python -m decision_index score --results <repo>/runs/di-smoke-slot-v0/results.jsonl
```

Order of work:

1. **Overlap check (needed before any number is reported).** Hash the states of the Index rows (the kit's `state` as our formatter sees it) and compare with the state hashes and 12-gram fingerprints of `train.parquet`, with the same code as `build_extra_eval`. A small script is not written yet; write it with a test. Report the share of Index requests per benchmark whose state or n-gram matches training. Do not drop rows from the Index; flag them.
2. Smoke run of 100 requests with the zero-shot model and one trained model (checks the adapter, memory and speed). Measure requests per second before the full run: the full suite is about 140k requests.
3. Full runs, one per model, primary setting `max_len` from the checkpoint (2,048), no other option changed. Second setting `max_len=4096` for the same models, reported separately.
4. `python -m decision_index score` per run. Record the public index, the area scores and the per-benchmark scores with coverage.
5. Latency: `mimir_decide.bench` on a sample of Index requests, against the 1,000 ms rule. Our adapter runs the questions of a request as one batch; it is not tuned.

## Metrics to record

Public index (chance-corrected, coverage-adjusted) and the raw index; the five area scores; per benchmark: score, coverage (answered / requests), unsupported count; the overlap share from step 1; median and p80 latency. Compare with the board's public numbers (`data/index.json` of the Space, same scorer) for the entrants closest in size.

## Decision rule

Written before running. No model selection is made on this benchmark.

- Slot and letter baseline: a difference of less than 1 point of the public index is a tie (my proposal; the board treats Full scores less than 0.9 points apart as sharing a rank). A larger difference is reported with its sign; it counts as evidence on [Q01](/questions/q01-clean-generalisation-evidence.md) only for benchmarks with overlap share below 5% and not flagged as seen by Mimir.
- Fine-tuned against zero-shot: judged per area, with coverage next to every score.
- A submission to the leaderboard (pull request, public weights) needs the owner's decision and is not part of this experiment.

## Limits

One seed per model. The public suite is 20% of the Full score; the private 80% cannot be run by us. Chance correction and weights are the board's, not ours. Many benchmarks are probably seen by Mimir by name (tasksource, FLAN); our mixture may contain their train splits. The letter model cannot answer questions with more than 26 options.

## Status and results

Status: **planned, deferred** by the owner on 2026-10-15 ("not sure if it's worth it"; the focus stays on Typed Decisions, [E14](/experiments/e14-typed-decisions-benchmark.md)). Do not start it without the owner's word. The adapter is written and tested offline; nothing is run on the real suite.

| Date | Run id / path | Config (overrides) | Result | Observation page |
|---|---|---|---|---|
| | | | | |
