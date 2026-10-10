---
type: Question
title: 'Q01: Where does clean generalisation evidence come from?'
description: Almost every usable source is flagged as seen by Mimir; we need data it cannot have seen.
status: open
priority: high
resolved_by: [e07, e03]
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Why it matters

A decision model is only convincing if it works on tasks and questions it was not trained or pre-exposed to.

## What we know

The name-level audit flags 316/316 tasksource, 7/7 bekko, 5/5 HelpSteer2 and 2/2 MASSIVE sources; only the 4 LocalLLaMA sources are unflagged ([O08](/observations/o08-audit-flag-rates.md)). LocalLLaMA is small, synthetic and templated. In-house sets (DaLA, GEC, Arena) are in Mimir's own policy too.

**2026-10-11:** [E12](/experiments/e12-unseen-label-sets.md) tests transfer to tasks with label sets unseen in fine-tuning; its two sources that Mimir's mix does not contain by name are the first test that is unseen for both.

## How to resolve

Options: (a) write or label new Danish decision items after the model was built; (b) use sources released after 2026-09-25 (v1.5 cut-off); (c) ask the Mimir team for a row-level check ([Q02](/questions/q02-row-level-overlap-with-mimir-data.md)); (d) report seen and not-flagged results side by side with `python -m mimir_decide.compare` ([E07](/experiments/e07-contamination-split-analysis.md)).

## Decision impact

Determines whether any generalisation claim can be made, and the choice of held-out tasks ([D15](/design/d15-held-out-task-massive-nb-no.md)).
