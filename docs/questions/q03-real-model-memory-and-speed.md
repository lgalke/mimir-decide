---
type: Question
title: 'Q03: Does the real 1.8B model train on one GPU?'
description: Memory, step time and throughput of Mimir v1.5 under our setup are unmeasured.
status: open
priority: high
resolved_by: [e02, e06]
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Why it matters

Everything so far ran on a tiny random model on CPU.

## What we know

Estimate only: fp32 weights ~7 GB, gradients and AdamW state for ~1B trainable parameters ~12 GB, plus activations at 2048 tokens with `L_bp_cycles [3,3]` and gradient checkpointing. `train.py` now logs `peak_mem_gb` and `ex_per_s`.

## How to resolve

Run [E02](/experiments/e02-smoke-run-real-model.md), then [E06](/experiments/e06-l-bp-cycles-ablation.md). Fallbacks: smaller `max_len`, `L_bp_cycles [0,3]`, 8-bit optimizer, bf16 weights.

## Decision impact

Sets batch size, max_len and how big the pilot can be ([D04](/design/d04-single-gpu-plain-loop.md), [D09](/design/d09-keep-checkpoint-l-bp-cycles.md), [D10](/design/d10-fp32-master-weights-bf16-autocast.md)).
