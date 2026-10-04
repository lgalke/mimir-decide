---
type: Question
title: 'Q05: Does CC-BY-SA data bind the model weights?'
description: Share-alike may or may not extend to a trained model.
status: open
priority: medium
resolved_by: []
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Why it matters

CC-BY-SA sources are on the allowlist ([D02](/design/d02-permissive-licences-only.md)).

## What we know

The manifest lists `share_alike_train_sources` (54 in the trial build). The legal position is unsettled.

## How to resolve

Ask the owner's institution or counsel; if needed, set CC-BY-SA out of the allowlist (`SHARE_ALIKE` in `licenses.py`) and rebuild.

## Decision impact

Affects the release licence of Mimir-Decide.
