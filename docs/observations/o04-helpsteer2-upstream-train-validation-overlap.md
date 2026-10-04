---
type: Observation
title: 'O04: HelpSteer2''s own train and validation overlap slightly'
description: 5 of 1,038 validation responses also appear in upstream train; 2 prompts do.
date: '2026-10-04'
confidence: high
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Observation

Measured directly on `nvidia/HelpSteer2`: 5/1,038 upstream validation responses occur in upstream train, 2/1,038 prompts. The exact-state hash misses these (prompt or response differs); the n-gram guard dropped 41 HelpSteer2 train decisions in the trial build.

## Interpretation

Even official splits leak a little; the guard is doing real work.

## Reproduce

Python: load train/validation with `datasets`, compare `norm_text(response)`; see `reports/ngram_leak_examples.jsonl`.

## Related

* [d12-ngram-fingerprint-guard-cross-source-only](/design/d12-ngram-fingerprint-guard-cross-source-only.md)
