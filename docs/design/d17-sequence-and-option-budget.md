---
type: Decision
title: 'D17: Sequence and option budget'
description: max_len 2048, middle truncation of the state, at most 26 options, long states dropped.
status: accepted
date: '2026-10-04'
decided_by: agent
tags: [data, training]
timestamp: '2026-10-04T00:00:00Z'
---

## Context

Mimir's context is 4,096; the slot model needs the whole question and all options.

## Decision

`max_len: 2048` (up to 4096). The state is cut from the middle with a marker; question and options are never cut (each option at most 160 tokens); rows where those alone do not fit are skipped and counted. States over 12,000 characters are dropped at build time. More than 26 options are subsampled keeping every option with target mass; if more than 26 have mass the row is dropped.

## Consequences

Bias against long-document tasks ([Q08](/questions/q08-long-state-drop-bias.md)); evaluation reports `n_skipped_too_long`.

## Revisit when

Memory allows longer sequences, or an important task is mostly long.

## Related

* [training-setup](/design/training-setup.md)
