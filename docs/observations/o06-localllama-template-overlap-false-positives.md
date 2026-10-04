---
type: Observation
title: 'O06: Within-source n-gram checks wrongly dropped 1,345 LocalLLaMA decisions'
description: Synthetic templated data repeats scenario text across its own train and test.
date: '2026-10-04'
confidence: high
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Observation

First version of the guard compared against all eval rows including the same source: 1,345 LocalLLaMA drops in the trial build. Restricting to eval rows of other sources reduced them to 0 and kept the real cross-collection catches.

## Interpretation

Trust the upstream split within a source; use exact and group guards there.

## Reproduce

`tests/test_build_leakage.py::test_ngram_guard_trusts_upstream_split_within_a_source`.

## Related

* [d12-ngram-fingerprint-guard-cross-source-only](/design/d12-ngram-fingerprint-guard-cross-source-only.md)
