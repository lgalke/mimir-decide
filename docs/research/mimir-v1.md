---
type: Model
title: DFM Mimir v1
description: Open 1B-parameter HRM-Text model from Danish Foundation Models, trained on permissible data; candidate base for a decision model.
resource: https://huggingface.co/danish-foundation-models/DFM-Mimir
tags: [mimir, hrm, danish, open-model]
timestamp: 2026-10-03T00:00:00Z
---

# DFM Mimir v1

> Superseded as the starting point by [Mimir v1.5](/research/mimir-v1-5.md) (2026-10-04).

- ~1B parameters, [HRM-Text](/research/hrm-text.md) architecture, trained from scratch.
- 161 datasets, ~70.5B tokens per epoch, permissible post-training data only.
- Context 4,096 tokens. License Apache 2.0.
- **Gemma4 tokenizer + chat template + PrefixLM attention mask are required.**
- Scores: English avg 69.0, Math & Code 64.1 (GSM8K 89.9), Danish 56.8 (state of the art for Danish). Beats HRM-Text 1B, competitive with Qwen 3.5 4B and Gemma 4 E2B.
- Built by SDU, Aarhus, Copenhagen, Alexandra Institute. A v2 of the paper adds memorization audits.

## Why it suits a decision model

Instruction-tuned, prefix-LM bidirectional encoding of the prompt (good for reading a state and questions), small, Apache-2.0, strong Danish. Weak point: 4k context limits long "state" documents.

Not verified: layer/hidden sizes, exact checkpoint formats, availability of a pre-post-training base checkpoint (only the abstract and HF summary were read).
