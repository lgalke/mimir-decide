---
type: Decision
title: Mimir to Decision Model, continual training plan
description: Proposed design for converting Mimir v1 into a Jev-like decision model. Our synthesis, untested.
status: superseded
date: '2026-10-03'
decided_by: agent (proposal)
tags: [plan, mimir, decision-model, proposal]
timestamp: 2026-10-03T00:00:00Z
---

# Proposed plan (hypothesis, not established fact)

> Partly superseded on 2026-10-04: base model is now [Mimir v1.5](/research/mimir-v1-5.md); the head design is in [heads-and-baseline](/design/heads-and-baseline.md); datasets and leakage rules are in [dataset-selection](/design/dataset-selection.md). Stage 1 (LLM-judge distillation) and the optional ACT early exit are not implemented. Deviations are listed in the [log](/log.md).

Jev's internals are undisclosed, so this is a design from public pieces: [decision models](/research/decision-models.md), [calibration training](/research/calibration-training.md), [Mimir](/research/mimir-v1.md), [HRM-Text](/research/hrm-text.md).

## 1. Prompt format

`[state] + [question k: type, options/rubric]` with PrefixLM so state and question are encoded bidirectionally. Each question is a separate sequence (or a separate query slot) so answers stay independent.

## 2. Output heads (replace the LM head for decisions)

- Noul: 1 logit, sigmoid.
- Choice: logits over option slots, softmax (options as pooled embeddings of option text, so the option set is open-vocabulary).
- Score: softmax over K ordered bins, expectation as score (ordinal loss optional).
- Readout from final H-state at a question token, or via learned query tokens. Keep the LM head to retain general ability and allow an LM-loss regulariser.

## 3. Continual-training stages

1. **Distil from an LLM judge:** generate (state, question, answer) labels with a strong LLM using sampled votes; use vote fractions as soft labels. Cheap, broad. Risk: inherits teacher bias.
2. **Ground-truth fine-tune** on labelled classification/NLI/moderation/routing datasets with cross-entropy/Brier.
3. **Calibration:** temperature scaling per primitive on held-out data; optionally RLCR-style Brier reward if any sampling stays in the loop.
4. **Replay:** mix in Mimir's original instruction data (and Danish) to limit forgetting.
5. **Optional:** ACT-style early exit on H-cycles for latency.

## 4. Evaluation

ECE, Brier, accuracy per primitive; agreement with human oracle; variance across paraphrases (Jev claims large variance reduction); latency vs. Laya (33-40 ms T4) and Jev (70-500 ms); Danish decisions; adversarial inputs; numbers/dates (Jev's weak spots); retained MMLU/GSM8K.

## 5. Risks

4k context; teacher bias; out-of-distribution calibration drift (Laya's calibration was not shown to transfer across domains); forgetting; licence compatibility of distillation labels with Mimir's "permissible data" ethos (use open teachers).

## Alternatives

Plain LoRA with a classification head would be a faster baseline; compare against it before investing in the HRM-specific readout.
