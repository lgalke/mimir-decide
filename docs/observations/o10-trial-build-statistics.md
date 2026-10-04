---
type: Observation
title: 'O10: Trial build statistics (--limit 4000)'
description: Counts and timing of the clean trial build used for smoke tests.
date: '2026-10-04'
confidence: medium
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Observation

Train 14,026 (noul 1,478, choice 6,832, score 5,716); validation 2,352; calibration 2,320; test 5,159; held-out 0 (the limit never reached nb-NO). 326 train sources, 230 without eval rows (limit effect), 54 CC-BY-SA sources. Eval hash set: 20,437 states, 16,268 groups, 292,128 fingerprints (3,807 boilerplate ignored). Build time 49 s. Train by collection: bekko 2,000, helpsteer2 3,959, localllama 4,000, massive 2,000, tasksource 2,067.

## Interpretation

The pipeline works end to end; numbers are not representative of a full build.

## Reproduce

`build_mixture --limit 4000`.

## Related

* [e01-full-mixture-build](/experiments/e01-full-mixture-build.md)
