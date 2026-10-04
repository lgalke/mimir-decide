---
type: Decision
title: 'D02: Permissive licences only'
description: Only train and evaluate on data whose licences are on a permissive allowlist; unknown or non-commercial is excluded.
status: accepted
date: '2026-10-04'
decided_by: user
tags: [licence, data]
timestamp: '2026-10-04T00:00:00Z'
---

## Context

Mimir is released on a 'permissible data' basis (Apache-2.0). Many sources in tasksource and bekko are unspecified, non-commercial or carry caveats.

## Decision

`mimir_decide/licenses.py` allows Apache-2.0, MIT, BSD, ISC, CC0, CC-BY 2.0/3.0/4.0, CC-BY-SA 3.0/4.0, ODC-BY, PDDL, Unlicense, AFL-3.0. Compound strings pass only if every part passes. Everything else is excluded and counted in `reports/license_exclusions.csv`.

## Consequences

- A large share of tasksource is lost (46% of the first 4,000 train rows read, [O01](/observations/o01-tasksource-licence-exclusion-rate.md)).
- bekko 'qualified' subsets are excluded by default ([D13](/design/d13-bekko-licence-and-test-handling.md)).
- CC-BY-SA data is allowed but listed in the manifest ([Q05](/questions/q05-share-alike-licence-on-weights.md)).

## Revisit when

The owner accepts non-commercial research-only use (then `check()` needs a mode switch), or a licence review admits 'qualified' subsets.

## Related

* [dataset-selection](/design/dataset-selection.md)
