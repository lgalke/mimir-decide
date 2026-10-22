---
type: Reference
title: How open Jev reproductions read out decisions, and what that says about frozen backbones
description: Model cards of four open reproductions (read 2026-10-16) show trained or LoRA-adapted backbones with encoder slots or per-option heads; none documents a frozen backbone. Notes what this means for our slot probe.
tags: [decision-model, readout, frozen-backbone, lora, related-work]
timestamp: 2026-10-17T00:00:00Z
---

# Readouts of open Jev reproductions and frozen backbones

Our first research ([Jev](/research/jev.md), [Laya](/research/laya.md), [calibration training](/research/calibration-training.md)) did not survey frozen-backbone approaches; the first plan only noted that 'plain LoRA with a classification head would be a faster baseline'. This page was added on 2026-10-17 after the E13 probes ([O23](/observations/o23-letter-probe-recovers-a-fifth-of-the-fine-tuning-gain.md), [O24](/observations/o24-slot-probe-extracts-little-from-a-frozen-backbone.md)). It rests on the model cards (and, for one model, its training script) listed below. They are second-hand and describe what the authors state, not what we reproduced.

## What four open reproductions do

| Model | Backbone | Readout | Backbone adapted? |
|---|---|---|---|
| [pngwn/system-one-qwen3.5-4b-scorer](https://huggingface.co/pngwn/system-one-qwen3.5-4b-scorer) (CC-BY-NC-4.0) | Qwen3.5-4B-Base, causal | One sequence per (state, question, option): `State:\n..\n\nQuestion:\n..\n\nOption:\n..`; a scalar sequence-classification head pools the last non-pad token; scores of one question are softmaxed. | LoRA (r=16, all linear projections). The script has no freezing option beyond that. |
| [heman10x/rlcd-modernbert-151m](https://huggingface.co/heman10x/rlcd-modernbert-151m) (Verdict) | ModernBERT-base, started from a GLiClass label-scoring model | 25 candidate slots in one pass (24 options plus an abstention slot). | Trained with cross-entropy plus Brier, then temperature scaling (T = 1.07). |
| [hotchpotch/bekko-system-one-v0-17m / 68m / 400m](https://huggingface.co/hotchpotch/bekko-system-one-v0-400m) | Ettin reranker encoders | Shared-prefix encoder with task-specific heads; candidates scored in parallel. | Trained; the base is already a cross-encoder. The card says the models fall far behind Jev on generalisation benchmarks. |
| [fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) | GLiNER2 encoder (340M, 1B) | Label set supplied at call time, scored in one pass. | Trained. |

Details from the Qwen scorer's script: option sets are capped at 16 during training (the correct option plus sampled negatives) but scored in full at evaluation; the question and option are never truncated, the state is cut from its end; temperature is fitted on validation. Its card states that calibration is the point: raw ECE 0.135, 0.044 after T = 1.75.

**Not found:** a documented frozen-backbone reproduction in these four cards. Other entrants of the [Decision Index](/research/decision-index.md) were not read.

## What this says for us

- Readouts at candidate or label positions (Verdict, GLiNER, Bekko) sit on **bidirectional encoders that were trained for that readout**. Our slot probe put untrained markers into a causal next-token model and trained only a 5,000-parameter head: 0.446 validation accuracy.
- The LM-based reproduction (Qwen scorer) reads the **last token of an option-specific sequence** and adapts the backbone with LoRA. That is the position a language model already uses for answers, as in our letter probe (0.703 frozen, 0.835 fully fine-tuned).
- After fine-tuning, our slot model and letter baseline tie on every measure ([O26](/observations/o26-unseen-label-sets-slot-and-baseline-tie.md), Q04). So the slot readout is not wrong after training, but nothing measured so far gives it an advantage; its argument is the missing limit of 26 options.

## Background not checked in this project

From general knowledge, not looked up here: linear probes of decoder-only LMs usually trail fine-tuning; fine-tuning that starts from a trained probe (LP-FT) is a known recipe against feature distortion. Do not cite as established for our setting without a source.

## Candidate experiments (not run, not decided)

- LoRA on the backbone plus the letter head, the closest to the published LoRA rows.
- A per-option sequence-classification head (one sequence per option, last-token pooling, softmax over options): no 26-option limit, K forward passes per question. Implemented as `model_type: scalar` ([D33](/design/d33-scalar-per-option-head.md)), run planned in [E17](/experiments/e17-scalar-head-against-letter-baseline.md).

## Related

* [D05](/design/d05-slot-readout-heads.md)
* [E13](/experiments/e13-head-only-baselines.md)
* [Decision Index](/research/decision-index.md)
