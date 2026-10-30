---
type: Decision
title: 'D38: Clef-style scoring head (model_type clef), single question per sequence'
description: A transformer scoring head after the design of Cloudflare's Clef, as used by Unsloth's decision-model recipe, on top of the Mimir backbone. Options are scored from pooled hidden states, a lexical vector from the LM-head rows, evidence routing over the whole sequence and a lexical prior. Joint decoding across questions is not included.
status: accepted
date: '2026-10-31'
decided_by: owner (implement it); implementation by agent from the published code
tags: [heads, clef, readout, unsloth]
timestamp: 2026-10-31T00:00:00Z
---

## Context

Unsloth's guide trains decision models with LoRA and "a small head, the same design as Cloudflare's Clef" (head learning rate 1e-4). Clef's code (`joint_schema_model.py`, Apache-2.0, on the Hub) shows the design; [E18](/experiments/e18-span-pooling-probe-and-fine-tuning.md) showed that a plain linear head over mean-pooled option text fails on unseen label sets ([O31](/observations/o31-span-probe-matches-the-letter-probe-in-distribution-but-fails-on-unseen-label-sets.md)). The owner asked to implement the Clef-style approach.

## Decision

- **Config:** `model_type: clef` (`configs/train_clef_v1.yaml`), classes `ClefHead` and `ClefHeadModel` in `mimir_decide/model.py`, formatter mode `clef` (the span prompt in the Gemma chat format, with exact token ranges of the question and of every option, `Encoded.qspan`, `span_start`/`span_end`).
- **Option description:** (a) the mean of the final hidden states over the option's tokens, (b) a lexical vector, the mean of the frozen LM-head rows of the option's tokens, (c) the mean state of the question text; projected to the head width.
- **Evidence routing:** `routing_layers` cross-attention layers (queries: the options; memory: all projected hidden states, padding masked).
- **Field vector:** question vector plus an attention-weighted summary of the options, the last-token state and a kind embedding (noul, choice, score; one shared head, not one per kind), refined by `layers` decoder layers that cross-attend to the memory.
- **Score:** `prior + sigmoid(gate) * joint`. The prior is a scaled cosine between the normalized lexical vector of the option and the normalized (question vector + last-token state), so it is the LM's own tendency to produce the option's words; the joint term is a scaled cosine between field and option plus an MLP of `[field, option, field*option, |field-option|]`. The scales and the gate are learned and start at 0 (scale 1, gate 0.5).
- **Size and training:** default width 512, 2 routing layers, 2 decoder layers, 8 heads, feed-forward 2048: 20.5M head parameters at hidden size 1536 (Clef uses width 1024 on a 5120-wide backbone). Head learning rate 1e-4 as in Unsloth's recipe (3e-4 suggested for a frozen probe). The head runs in fp32. Backbone: trained fully or frozen (`freeze_backbone`); LoRA, which Unsloth uses, is not implemented here.
- Loss, EMD term for score, calibration and evaluation are shared with the other models.

## Not included (differences from Clef)

- **Joint decoding across the questions of a state.** Clef puts all questions of a record into one sequence and lets them interact. Our pipeline treats every question as its own decision, so the decoder here sees one field. Joint decoding needs batching by `case_id` and a record-level loss and evaluation.
- Clef's prompt format (system prompt, `FIELD i`, `OPTION k: {json}`); ours is the chat-format span prompt.
- Vision inputs and LoRA.

## Consequences

- One forward pass per decision; the cost of the backbone is that of the letter model, the head adds a cross-attention over the sequence (small).
- Tested offline: exact spans, the prior equals the manual cosine of the LM-head rows (head gate closed), finite logits for all kinds and option counts with invalid slots masked, a frozen probe trains only the head, a full step trains the backbone too and keeps the LM head frozen, padding invariance, a checkpoint round trip (network test) and train, calibrate and evaluate on a tiny model for the full model and the probe. **Not run on the real model.**
- The implementation follows the published Clef code from reading it; it is not a checked copy of Unsloth's training code.

## Revisit when

[E19](/experiments/e19-clef-style-head.md) shows a difference to the other readouts, or joint decoding across questions is wanted.

## Related

* [D35](/design/d35-span-pooling-readout-with-per-kind-heads.md)
* [D05](/design/d05-slot-readout-heads.md)
* [frozen-and-adapted-backbones](/research/frozen-and-adapted-backbones.md)
