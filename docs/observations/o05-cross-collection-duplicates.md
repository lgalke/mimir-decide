---
type: Observation
title: 'O05: The same tasks arrive via several collections'
description: bekko and tasksource carry twins of cladder and corr2cause; aqua_rat overlaps math_qa.
date: '2026-10-04'
confidence: high
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Observation

N-gram guard examples (trial build): `tasksource/cladder` against `bekko/cladder`, `tasksource/corr2cause` against `bekko/corr2cause`, `bekko/aqua_rat` (13 hits) against `tasksource/math_qa`, `tasksource/spartqa-yn` against `tasksource/spartqa-mchoice`. Renderings differ, so exact hashes do not match.

## Interpretation

Cross-collection leakage is real and rendering-independent checks are needed.

## Reproduce

`reports/ngram_leak_examples.jsonl` in the trial build.

## Related

* [d12-ngram-fingerprint-guard-cross-source-only](/design/d12-ngram-fingerprint-guard-cross-source-only.md)
