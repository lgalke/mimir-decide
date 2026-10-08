---
type: Question
title: 'Q04: Does the slot readout beat the letter baseline on the real model?'
description: Only a toy synthetic check exists.
status: open
priority: high
resolved_by: [e03]
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Why it matters

This is the central design bet ([D05](/design/d05-slot-readout-heads.md)).

## What we know

On a 1-layer toy task the slot model learned shuffled-option decisions in ~120 steps while the letter baseline needed fixed order ([O09](/observations/o09-toy-slot-vs-letter.md)). That says nothing about the real model, where the letter format is natively familiar from instruction tuning.

**First pilot result (2026-10-09, one seed):** a tie within 0.004 on accuracy, NLL and Brier, validation and held-out ([O14](/observations/o14-first-pilot-results-slot-and-baseline-tie.md)). This is not an answer yet: almost all data is familiar to Mimir, and there is no zero-shot or second-seed reference. The open parts are robustness to option order ([E05](/experiments/e05-option-order-robustness.md)), more than 26 options, and truly unseen tasks.

## How to resolve

[E03](/experiments/e03-slot-vs-letter-pilot.md) with at least 2 seeds, reported with `compare.py`.

## Decision impact

If the baseline matches the slot model, the simpler baseline wins and the head work can be dropped.
