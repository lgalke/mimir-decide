---
type: Decision
title: 'D33: Scalar per-option head as a third readout (model_type scalar)'
description: Every option is its own sequence; a LayerNorm and a linear layer on the final state at the last token give one score; the scores of a decision are softmaxed. Training samples negatives for choice decisions with a sampled-softmax correction.
status: accepted
date: '2026-10-22'
decided_by: owner (wanted the scalar head); implementation by agent
tags: [heads, scalar, readout]
timestamp: 2026-10-22T00:00:00Z
---

## Context

The owner finds the scalar per-option head conceptually most appealing: it is the standard reward-model or cross-encoder design and the one used by the LM-based open reproduction ([frozen-and-adapted-backbones](/research/frozen-and-adapted-backbones.md)). The slot head ([D05](/design/d05-slot-readout-heads.md)) and the letter baseline ([D06](/design/d06-letter-logit-baseline.md)) tie on every test so far, so a third readout is not expected to win on quality; its value is the missing 26-option limit and the standard design.

## Decision

- **Config:** `model_type: scalar` (`configs/train_scalar_v1.yaml`), code in `OptionScorerModel` (`mimir_decide/model.py`) and the formatter mode `scalar` (`mimir_decide/formatting.py`).
- **Input:** one row per option, in the Gemma chat format ([D34](/design/d34-chat-format-for-slot-and-scalar.md)): `<bos><|turn>user\nState:\n{state}\n\nQuestion ({kind}):\n{question}\n\nOption:\n{option}<turn|>\n<|turn>model\n` (the plain variant `<bos>State:\n…` exists as `chat_format: false`). The option is the end of the row and is never cut apart from the 160-token limit; the state is cut from the middle when needed, to the same length for all rows of a decision. The rows are bidirectional PrefixLM blocks like all other formats.
- **Readout:** the final hidden state of the backbone at the last real token of each row (the final token of the generation prompt in the chat format) -> `LayerNorm -> Linear(1536, 1)` -> times `exp(log_scale[kind])` (a learned scale per kind, as in the slot model) -> scattered into `[B, K]` logits -> softmax over the options of the decision. Loss, EMD term for score, calibration, evaluation and the Typed Decisions scorer are unchanged.
- **Sampled negatives (training, choice decisions only):** keep every option with target mass and add random zero-mass options up to `max_train_options` (default 6, at least one negative); the sampled-softmax correction `log(#negatives / #kept)` is added to the logits of kept negatives in the loss so that the normaliser stays unbiased. Noul (two options) and score decisions (ordered bins, the EMD term needs all of them) are never sampled. Evaluation always uses all options.
- **Micro-batch:** `batch_size` counts decisions, a decision is K rows; the scalar config uses `batch_size: 2`, `grad_accum: 16` (effective 32) as a starting point, to be checked with `mem_probe --options 6`.

## Consequences

- **Property by construction:** the score of an option does not depend on the order of the options or on the other options (tested), so option-order sensitivity is zero for this readout. The other options influence only the softmax.
- **Cost:** about K forward passes per decision instead of one (the state cannot be shared under the bidirectional PrefixLM), bounded in training by `max_train_options` for choice; noul costs 2 rows, a score decision 5 to 11. Throughput is not measured.
- **No option limit** from letters; the length limit of the context still applies.
- Tested offline: formatting, row packing, negative sampling, order and padding invariance, a training step, a checkpoint round trip (network test) and train, calibrate and evaluate on a tiny random model. **Not run on the real model or the GPU.**

## Revisit when

The GPU smoke test shows an unacceptable cost per example, or E17 shows a quality difference to the letter baseline.

## Related

* [E17](/experiments/e17-scalar-head-against-letter-baseline.md)
* [D05](/design/d05-slot-readout-heads.md)
* [D06](/design/d06-letter-logit-baseline.md)
