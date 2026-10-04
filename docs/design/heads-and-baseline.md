---
type: Decision
title: Output heads strategy and the LM-letter baseline
description: One shared open-vocabulary slot readout for Noul/Choice/Score instead of per-task heads, plus the letter-logit baseline to compare against.
status: accepted
date: '2026-10-04'
decided_by: user (accepted proposal)
tags: [heads, baseline, slot-readout, decision-model]
timestamp: 2026-10-04T00:00:00Z
---

# Output heads strategy and the LM-letter baseline

Scope of this page: only the head design and the baseline. Datasets are in [dataset-selection](/design/dataset-selection.md); runs are in [training-setup](/design/training-setup.md).

## Decision: one shared slot readout, not one head per task

A fixed head per task or label set cannot do what [Jev](/research/jev.md) does: questions and options are supplied at inference time ([decision models](/research/decision-models.md)). So the three primitives share one readout and differ only in loss and calibration.

- **Slot readout.** Options go into the prompt as marked slots. The model reads one scalar logit per option from the final H-state at that option's marker token. A softmax over the slots of one decision gives the distribution. PrefixLM lets the state, the question and all options see each other.
- **Choice** is the softmax over its slots.
- **Noul** is a Choice with two slots (yes/no or the authored true/false definitions). The softmax over two slots equals a sigmoid of their difference.
- **Score** is a Choice over ordered bins. Training adds a squared-CDF (EMD²) penalty with weight 0.5 so that mass placed far from the true bin costs more than mass placed next to it.
- **Per-kind scale and temperature.** A learned scale per kind (noul, choice, score) is applied to the logits during training, and one temperature per kind is fitted afterwards on held-out calibration data ([calibration training](/research/calibration-training.md)). A per-kind bias would be useless because the softmax is shift-invariant.
- **Independence.** Each decision is its own sequence, so answers cannot influence each other (Jev evaluates questions independently).
- **Option order.** Order is shuffled during training, with the target permuted accordingly. Score options are only ever presented in ascending or fully reversed order, which keeps the ordinal loss valid. Evaluation uses the given order.
- **Language head kept.** The LM head stays in the checkpoint, frozen, so the model can still be used generatively.

Implementation notes and deviations from the first plan are in the [log](/log.md): the marker is a learned vector inserted at placeholder positions rather than a new vocabulary token.

## Baseline: letter logits from the unchanged language-model head

The baseline answers whether the new readout is worth having. It uses the same data, backbone, loss, calibration and metrics, and differs only in how the answer is read out.

- Options are labelled `A.`, `B.`, … inside a Gemma chat turn, followed by an instruction to answer with the letter.
- The prediction is the LM-head logits of the letter tokens at the last position, softmaxed over the options present.
- The prompt was checked to be identical to the tokenizer's chat-template encoding of the same text.
- Options are capped at 26 so each has a letter.
- The LM head and input embeddings are frozen by default (configurable), as in the slot model.

## What we know so far

A small synthetic test (3 options, label given by a keyword in the state, 1-layer toy HRM) shows the expected difference: with shuffled options the slot model's loss fell to about a third within 120 steps, while the letter baseline stayed near chance (1.12 to 0.81 after 400 steps) and only learned quickly with a fixed option order. This is a toy mechanism check, not evidence about the real model. The real comparison is the pilot ([training-setup](/design/training-setup.md)).
