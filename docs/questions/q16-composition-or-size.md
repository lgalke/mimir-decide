---
type: Question
title: 'Q16: Do more data, more diverse data or a longer context help most?'
description: Mixture v1 changes size, composition and context length. Which of them moves quality on unseen label sets and the Typed Decisions benchmark?
status: open
priority: high
resolved_by: [e16]
tags: [mixture, diversity]
timestamp: 2026-10-21T00:00:00Z
---

## Why it matters

Training time is the main cost. If composition matters more than size, a smaller diverse mixture saves hours; if size matters, the plan should spend compute on the B200.

## What we know

Nothing yet. The pilot mixture (371k decisions, 366 sources, capped at 2,000 per source) gives the baseline. [E12](/experiments/e12-unseen-label-sets.md) shows that fine-tuning gains 8 to 9 points on unseen label sets, and slot and letter baseline tie.

**Update 2026-10-24 ([O29](/observations/o29-mixture-v1-1m-letter-baseline-results.md)):** the 1M letter baseline (size, composition, 4096 context and 31k steps together) gains 1.0 point of validation accuracy and 0.03 NLL over the pilot but loses 1.7 points on the 32 unseen-label-set sources (E12), and is level on Typed Decisions accuracy with 7% lower KL. More data in the same sources helps in-distribution and hurts transfer to new label sets in this one-seed comparison; the cause (steps, composition, learning-rate scaling) is open.

## How to resolve

[E16](/experiments/e16-mixture-v1-training.md): v1 at v0's size against v0, then a larger v1, with the evaluation frozen.

## Decision impact

Chooses the size of the next main training run and whether the family and kind targets of [D31](/design/d31-mixture-v1-size-and-diversity.md) stay.
