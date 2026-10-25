---
type: Decision
title: 'D35: Span-pooling readout with one head per kind (model_type span)'
description: All options in one sequence; the representation of an option is the mean of the final hidden states over its own text; one linear head per kind turns it into a score. Our reading of the approach of the owner's PhD student; frozen-backbone probe and full fine-tuning are both supported.
status: accepted
date: '2026-10-25'
decided_by: owner (run it); implementation by agent
tags: [heads, readout, span, per-kind]
timestamp: 2026-10-25T00:00:00Z
---

## Context

The owner's PhD student got a good result with per-kind heads on the average activation of each option's string, apparently from one forward pass. Details are not known (which backbone, whether only the heads or the whole model were trained, data, prompt format). The slot readout reads a marker token that the pretrained model has no reason to make informative (O24); mean pooling reads the option's own tokens.

## Decision

- **Config:** `model_type: span` (`configs/train_span_v1.yaml`), class `SpanPoolModel` (`mimir_decide/model.py`), formatter mode `span` (`mimir_decide/formatting.py`).
- **Input:** the slot prompt without markers, in the Gemma chat format ([D34](/design/d34-chat-format-for-slot-and-scalar.md), `chat_format` default true): `<bos><|turn>user\nState:..\n\nQuestion (kind):..\n\nOptions:\n{option 1}\n{option 2}…<turn|>\n<|turn>model\n`. One sequence per decision, all real tokens one bidirectional PrefixLM block, so every option sees the state, the question and the other options. Options are cut at 160 tokens as before.
- **Readout:** for option j the mean of the final hidden states over the tokens of its text (the `Encoded.spans` range, separator and newline excluded), then `LayerNorm -> Linear(1536, 1)` with **one head per kind** (noul, choice, score; `per_kind_heads: true`). With `per_kind_heads: false` one shared head is scaled by a learned factor per kind, as in the slot model. Loss, EMD term, calibration and evaluation are shared.
- **Frozen backbone:** `freeze_backbone: true` trains only the heads (no backward pass through the backbone; the checkpoint stores the heads and loads the base model from the hub), as for the E13 probes.
- Order of the options influences the scores here (the options are in one sequence), unlike for the scalar head.

## Consequences

- One forward pass per decision, so the cost is that of the letter baseline and the slot model, not that of the scalar head.
- The per-kind heads are about 4,600 parameters each; they add no memory.
- Tested offline: spans cover exactly the option text and move with a permutation, the pooled vector equals the manual mean, kinds use their own head, the frozen probe trains only the heads, a full training step, a checkpoint round trip (network test) and train, calibrate and evaluate for both modes on a tiny random model. **Not run on the real model.**
- Whether this matches the student's method is unknown; compare only after her design is known ([E18](/experiments/e18-span-pooling-probe-and-fine-tuning.md)).

## Revisit when

E18 shows a difference to the letter baseline, or the student's description differs from this reading.

## Related

* [D05](/design/d05-slot-readout-heads.md)
* [D33](/design/d33-scalar-per-option-head.md)
* [O24](/observations/o24-slot-probe-extracts-little-from-a-frozen-backbone.md)
