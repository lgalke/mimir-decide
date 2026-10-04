---
type: Question
title: 'Q02: Which of our eval rows did Mimir see?'
description: Row-level overlap cannot be computed from public artifacts because the DFM10 base data is private.
status: open
priority: medium
resolved_by: []
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Why it matters

Name-level flags say a collection was in the mixture, not whether our eval rows were.

## What we know

v1.5 publishes `training_data_manifest.json` and `dfm11_sampling_policy.yaml` (193 prefixes) and 11 public addition packages (tool use, maths, koolbardi, fineinstructions). The base `tokenized_dfm10` is not public.

## How to resolve

Ask the Mimir team to hash our `validation.parquet`, `calib.parquet`, `heldout.parquet` and `test.parquet` states against their tokenized data and return the overlap counts per source. Optionally hash the public addition packages ourselves.

## Decision impact

Would let us drop or separately report contaminated eval rows instead of whole sources.
