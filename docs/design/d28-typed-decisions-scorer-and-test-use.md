---
type: Decision
title: 'D28: Typed Decisions benchmark: own scorer, shared predictions, one test pass per model'
description: The benchmark's test split is used once per model to compare approaches; the scorer's definitions are fixed and stated; per-decision probabilities are saved so any approach can be rescored.
status: accepted
date: '2026-10-12'
decided_by: owner (priority and use of the test split); definitions by agent
tags: [evaluation, benchmark, test-split]
timestamp: 2026-10-12T00:00:00Z
---

## Context

A PhD student of the owner works on a Mimir-Decide approach and evaluates on the test split of LocalLLaMA/typed-decisions. The owner made a comparison on that data the top priority. Our own rule ([D11](/design/d11-split-isolation-and-test-access.md)) allows rare, logged test access.

## Decision

- Our models are scored on the full test split (2,000 decisions) with `mimir_decide.typed_decisions`; the access needs `--final` and is logged.
- **One pass per model:** the call computes the logits once and writes the calibrated and the raw version. Nothing is selected on these scores.
- The scorer's definitions are fixed and written into every result file ([O21](/observations/o21-benchmark-scorer-matches-the-published-uniform-row-only-for-kl-and-brier.md)). The per-decision probabilities are saved (`*_predictions.json`) so that other approaches can be rescored with the same code.
- Our rows are labelled 'trained on `train` and other data' (our mixture contains the benchmark's train split, planned about 90% of it; after the build's drops at most about 56%, see [O25](/observations/o25-full-mixture-build-counts-and-drops.md)); the zero-shot model is the only 'general, zero-shot' row.
- Nothing is submitted to the benchmark's discussion board without the owner's decision.

## Consequences

- The test split of this external benchmark is used for reporting, once per model. Later models (the probes) are scored the same way.
- The numbers are not comparable with the README's rows in accuracy and ECE, and our mode (one question per request) differs from theirs.

## Revisit when

Anna's scorer or the benchmark's official definitions become available (then adopt them), or the benchmark is updated.

## Related

* [D11](/design/d11-split-isolation-and-test-access.md)
* [E14](/experiments/e14-typed-decisions-benchmark.md)
