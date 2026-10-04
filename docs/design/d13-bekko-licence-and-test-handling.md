---
type: Decision
title: 'D13: bekko: strict licence status, hash-only test'
description: Use only 'verified' upstream licences; hash bekko test rows but never write them.
status: accepted
date: '2026-10-04'
decided_by: agent
tags: [data, licence, leakage]
timestamp: '2026-10-04T00:00:00Z'
---

## Context

bekko licences are per upstream dataset with review status 'verified' or 'qualified' (data-specific caveats). Its card says membership is defined only by the root training manifest.

## Decision

`accept_qualified: false`. Membership from `training-manifest.json` (train list; `evaluation` list with split=validation|test); quarantine list honoured (currently empty). `write_test: false`: bekko test rows are used only for the leakage guard.

## Consequences

bekko shrinks a lot ([Q06](/questions/q06-bekko-coverage-after-strict-licences.md)). 12 of 17 excluded subsets in the trial build were 'qualified'.

## Revisit when

A review admits specific qualified subsets (set `accept_qualified` or add an allowlist).

## Related

* [dataset-selection](/design/dataset-selection.md)
