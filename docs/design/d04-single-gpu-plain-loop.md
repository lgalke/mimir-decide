---
type: Decision
title: 'D04: Single GPU node, plain PyTorch loop'
description: No SLURM, no multi-GPU; a simple single-process training loop.
status: accepted
date: '2026-10-04'
decided_by: user
tags: [compute, training]
timestamp: '2026-10-04T00:00:00Z'
---

## Context

The owner will run on a single GPU node and hand the session to an agent on a GPU cluster (opencode).

## Decision

`train.py` is a plain loop (AdamW, cosine schedule, grad accumulation, optional gradient checkpointing). No DDP/FSDP and no launcher scripts.

## Consequences

- Memory-bound by one device ([Q03](/questions/q03-real-model-memory-and-speed.md)).
- Throughput is whatever one GPU gives; keep `max_total_train` modest.

## Revisit when

One GPU cannot hold the model plus optimizer, or the pilot is too slow to iterate.

## Related

* [training-setup](/design/training-setup.md)
