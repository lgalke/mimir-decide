---
type: Decision
title: 'D36: Optional weight average (EMA) of the trainable parameters, evaluated next to the training weights'
description: Option ema_decay keeps an exponential average of the trainable parameters, evaluates it at every validation point, and writes best_ema/ and final_ema/ checkpoints. Off by default; the training trajectory is unchanged.
status: accepted
date: '2026-10-25'
decided_by: owner (implement it); implementation by agent
tags: [training, ema, checkpoints]
timestamp: 2026-10-25T00:00:00Z
---

## Context

Mimir's pretraining used an EMA of the weights with decay 0.9999, a constant learning rate after a 2,000-step warm-up and weight decay 0.1 (`config/cfg_pretrain.yaml` of HRM-Text); the released weights are the EMA export. Our fine-tuning used none of this: cosine decay to 10% of the peak learning rate, weight decay 0.01, the best checkpoint by validation NLL. With an effective batch of 32 decisions the gradient noise is high, so an average of the weights may help a little. This is untested.

## Decision

- **Config:** `ema_decay: null` (off) in the v1 training configs; for example `ema_decay=0.999`. A decay of 0.9999 would average over about 10,000 steps, which is the length of a whole fine-tuning run; 0.999 (about 1,000 steps) is the suggested value, not tuned.
- **What is averaged:** all trainable parameters (backbone, heads, scales, marker), kept in fp32 (extra memory about 4 GB for the 981M backbone parameters; requires `param_dtype: float32`). The decay warms up from about 0.1 to the target (`min(decay, (1+t)/(10+t))`).
- **Evaluation and checkpoints:** at every validation point the averaged weights are evaluated on the same 2,000 decisions as the training weights (log record `eval_ema`, next to `eval`), and `best_ema/` (by their own validation NLL) and, at the end, `final_ema/` are written next to `best/` and `final/`. Evaluation, calibration and the benchmark scorer use them with `--checkpoint best_ema` or `final_ema`. The averaged weights are swapped in without a copy and swapped out again, so training is unchanged.
- **Plot:** `python -m mimir_decide.plot_curves ... --ema` adds the averaged-weights curves.

## Consequences

- Training with the option on follows the same trajectory as without (the average does not feed back); the cost is about 4 GB and a second validation pass per evaluation. A tiny-model run gives the same raw validation NLL with and without it.
- It gives a direct within-run answer whether averaging helps: compare `eval_ema` with `eval` at each step.
- Not tested on the real model; the 6-step smoke on a tiny model wrote `eval_ema` lines and both EMA checkpoints, and the EMA checkpoint evaluates.
- The other differences to Mimir's recipe (weight decay 0.1, constant learning rate) are not changed.

## Revisit when

A run with the option shows an effect of averaging; then the decay and the use of `best_ema` for the final model are decided on validation data.

## Related

* [D30](/design/d30-galke-poech-lr-dividers.md)
* [training-setup](/design/training-setup.md)
* [O28](/observations/o28-mimir-basics-check.md)
