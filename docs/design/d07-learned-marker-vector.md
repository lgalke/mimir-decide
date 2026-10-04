---
type: Decision
title: 'D07: Option marker is a learned vector, not a new token'
description: Placeholder positions in the prompt get a learned embedding instead of an added vocabulary token.
status: accepted
date: '2026-10-04'
decided_by: agent
tags: [model, implementation]
timestamp: '2026-10-04T00:00:00Z'
---

## Context

The first plan added a `<|opt|>` token and resized the embeddings. That means touching a 262k x 1536 matrix, and freezing everything except one new row needs gradient masks, which AdamW weight decay would undo.

## Decision

The prompt contains the existing `<unused0>` token at each option; `SlotDecisionModel` replaces those positions with a `marker` parameter (mean embedding at init, unscaled space) before the backbone, which applies `embedding_scale` itself. Embeddings stay frozen.

## Consequences

- No resize; checkpoints store the marker in `extra.pt` and `marker_id` in `decision_meta.json`.
- Markers are identified by an explicit mask, not by token id.

## Revisit when

A generative use of the marker token is wanted.

## Related

* [heads-and-baseline](/design/heads-and-baseline.md)
