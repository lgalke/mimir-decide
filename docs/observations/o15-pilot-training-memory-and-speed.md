---
type: Observation
title: 'O15: Pilot training fits in 33.6 GB and runs at 6.15 examples per second'
description: Memory and speed of the first real training run (slot model, default settings) on an NVIDIA RTX PRO 6000 Blackwell Server Edition.
date: '2026-10-09'
confidence: medium
tags: [pilot, memory, speed, e02]
timestamp: 2026-10-09T00:00:00Z
---

## Observation

Source: 21 consecutive lines of `train_log.jsonl` (steps 9,570 to 9,770) pasted by the owner during the first training run of `scripts/run_pilot.sh`, which is the slot model. The owner did not report overrides, so the defaults of `configs/train_slot.yaml` are assumed: `L_bp_cycles [3,3]` (checkpoint default), batch 8 with 4 accumulation steps, `max_len` 2048, fp32 weights with bf16 autocast, gradient checkpointing on.

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition.
- `peak_mem_gb`: 33.61, constant over the window.
- `ex_per_s`: 6.15 to 6.16. At step 9,770 the elapsed time was 50,740 s, so about 5.2 s per optimizer step.
- `skipped`: 0 (no decision was too long).
- Learning rate at step 9,770: 1.55e-6, close to the schedule floor of 1e-6.
- Inferred, not read from the manifest: the cosine schedule puts step 9,770 at about 84% of training, so the run has about 11.6k optimizer steps, or about 370k training decisions, and about 17 hours per model. The `mixture_manifest.json` must confirm the count.
- The logged `loss` is the loss of the last micro-batch (8 decisions) and is noisy: 0.11 to 1.12, mean 0.39 over the 21 lines. `grad_norm` (before clipping) is 4 to 25 while `grad_clip` is 1.0, so clipping was active at every step.

## Interpretation

The real model trains on one GPU with the checkpoint's own gradient setting, with room to spare if the card has 96 GB (not checked). This answers [Q03](/questions/q03-real-model-memory-and-speed.md) for this GPU. A larger batch is possible but would change the training regime. The pilot needs about 35 hours for both models. Constant clipping suggests trying a higher `grad_clip` in a later run; that was not tested.

## Reproduce

`grep peak_mem_gb $RUNS/slot-v0/train_log.jsonl | tail -1`

## Related

* [E02](/experiments/e02-smoke-run-real-model.md)
* [D09](/design/d09-keep-checkpoint-l-bp-cycles.md)
* [D10](/design/d10-fp32-master-weights-bf16-autocast.md)
