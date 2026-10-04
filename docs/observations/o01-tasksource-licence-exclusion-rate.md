---
type: Observation
title: 'O01: 46% of the first 4,000 tasksource train rows are excluded by licence'
description: 'Trial build: unknown, non-commercial or non-allowlisted licences remove nearly half of sampled tasksource rows.'
date: '2026-10-04'
confidence: medium
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Observation

Trial build `--limit 4000` (rows are ordered by source name, so not a random sample): 1,249 `unknown_license`, 397 `non_commercial`, 190 `not_on_allowlist` out of 4,000 rows read = 1,836 (46%) excluded. Another 275 had empty text. The first 3,000 streamed rows had 746 'unspecified', 163 'other', 84 `cc-by-nc-4.0`.

## Interpretation

Licence strings are often compound or free-text (`CC BY 4.0 (DPI)`, `cc-by-4.0, CC BY 4.0 (DPI)`), which the normaliser now handles.

## Reproduce

`python -m mimir_decide.build_mixture --limit 4000 --output_dir <dir>`; see `<dir>/mixture_manifest.json` drops and `reports/license_exclusions.csv`.

## Related

* [d02-permissive-licences-only](/design/d02-permissive-licences-only.md)
* [q06-bekko-coverage-after-strict-licences](/questions/q06-bekko-coverage-after-strict-licences.md)
