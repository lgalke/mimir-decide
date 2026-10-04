---
type: Observation
title: 'O03: Dataset cards and the data disagreed in several places'
description: Schemas and split labels had to be verified by reading real rows.
date: '2026-10-04'
confidence: high
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Observation

tasksource: noul rows have empty `options` and a single target = P(true); the validation split's `split` column says `dev` (ids say `validation`), which made a strict equality check drop all 15,000 validation rows. LocalLLaMA: the real columns are JSON strings (`state`, `questions`, `gold`), some noul questions have no `criteria`; a card-based first converter yielded nothing. MASSIVE: HF repo is a loading script. bekko: licences per upstream dataset in `sources.json`; ranking decisions have documents instead of criteria. HelpSteer2: no test split.

## Interpretation

Never trust a card summary for converters; check rows and drop counts. The builder now lists train sources without eval rows and warns about unmatched held-out patterns.

## Reproduce

`tests/test_converters.py` encodes the verified schemas; compare with HF dataset viewers.

## Related

* [d14-split-carving-where-upstream-has-no-clean-split](/design/d14-split-carving-where-upstream-has-no-clean-split.md)
* [dataset-selection](/design/dataset-selection.md)
