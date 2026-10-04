---
type: Decision
title: 'D10: fp32 master weights with bf16 autocast'
description: Parameters in float32, forward in bf16 on CUDA.
status: accepted
date: '2026-10-04'
decided_by: agent
tags: [training, memory]
timestamp: '2026-10-04T00:00:00Z'
---

## Context

Pure bf16 weights lose updates at a learning rate of 1e-5.

## Decision

`param_dtype: float32`, `autocast: true` on CUDA. `param_dtype: bfloat16` exists for smoke runs on small machines only.

## Consequences

About 7 GB of weights plus gradients and AdamW state for ~1B trainable parameters. Not measured on a GPU ([Q03](/questions/q03-real-model-memory-and-speed.md)).

## Revisit when

Memory forces bf16 weights (then consider a Kahan-style or 8-bit optimizer).

## Related

* [training-setup](/design/training-setup.md)
