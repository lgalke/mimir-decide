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

Update 2026-10-21 ([D31](/design/d31-mixture-v1-size-and-diversity.md)): mixture v1 trains at `max_len: 4096` and admits train states up to 16,000 characters (`train_max_state_chars`); the eval files keep the 12,000-character filter, so they stay identical to v0. The 160-token option limit and the middle cut of longer states stay.

## Revisit when

Memory allows longer sequences, or an important task is mostly long.

## Related

* [training-setup](/design/training-setup.md)
