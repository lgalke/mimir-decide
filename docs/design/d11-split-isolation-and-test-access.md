---
type: Decision
title: 'D11: Split isolation and test-access protocol'
description: Train on train splits only; eval data of every source is hashed first; test is touched only with --final and logged.
status: accepted
date: '2026-10-04'
decided_by: agent (from the owner's requirement)
tags: [leakage, evaluation, process]
timestamp: '2026-10-04T00:00:00Z'
---

## Context

The owner required that we never train on test sets, only train splits.

## Decision

1. Converters read each dataset's own splits; split labels normalised (`dev` is validation). 2. The builder hashes all validation/test (any licence) before reading train. 3. Guards: exact state hash, group id, cross-source n-gram fingerprints ([D12](/design/d12-ngram-fingerprint-guard-cross-source-only.md)). 4. Final assertions on written files fail the build. 5. `train.py` refuses `eval/`, `heldout_tasks/`, `test*`, `heldout*` paths. 6. `evaluate.py --split test` requires `--final` and logs to `eval/test_access_log.jsonl`. 7. Calibration data is half of validation.

## Consequences

Each guard was disabled in turn and the tests failed, so the tests do detect regressions. Agents must not read or evaluate on test casually: test results should be rare and recorded.

## Revisit when

A new data source is added: it needs its own split mapping and a test.

## Related

* [dataset-selection](/design/dataset-selection.md)
* [training-setup](/design/training-setup.md)
