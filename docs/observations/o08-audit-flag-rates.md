---
type: Observation
title: 'O08: Name-level audit flags almost everything'
description: 316/316 tasksource, 7/7 bekko, 5/5 HelpSteer2, 2/2 MASSIVE; only LocalLLaMA (0/4) is unflagged.
date: '2026-10-04'
confidence: medium
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Observation

Trial build audit. HelpSteer2 and MASSIVE are flagged through an alias rule: tasksource contains `HelpSteer2/*` and `multilingual/massive`, and Mimir saw `tasksource__`.

## Interpretation

No clean held-out evidence among public sources; my first suggestion that `massive/nb-NO` was clean was wrong. Absence of a flag is not proof of no overlap.

## Reproduce

`<mixture>/reports/mimir_overlap.md`.

## Related

* [d15-held-out-task-massive-nb-no](/design/d15-held-out-task-massive-nb-no.md)
* [q01-clean-generalisation-evidence](/questions/q01-clean-generalisation-evidence.md)
