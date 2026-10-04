---
type: Question
title: 'Q09: Should we add LLM-judge distillation?'
description: Stage 1 of the first plan is not implemented.
status: open
priority: medium
resolved_by: []
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Why it matters

Open-vocabulary breadth could come from soft labels of a strong open LLM over new (state, question, options).

## What we know

[First plan](/design/mimir-to-decision-model-plan.md). Teacher bias and licence of teacher outputs are concerns; use open teachers.

## How to resolve

Decide after [E03](/experiments/e03-slot-vs-letter-pilot.md): if held-out transfer is poor, generate teacher labels with vote fractions as soft targets.

## Decision impact

Breadth of questions the model handles.
