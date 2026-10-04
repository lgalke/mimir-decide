---
type: Model
title: DFM Mimir v1.5
description: Final epoch-10 EMA checkpoint of the XL HRM-Text lineage; the chosen starting point for Mimir-Decide.
resource: https://huggingface.co/danish-foundation-models/DFM-Mimir-v1.5
tags: [mimir, hrm, danish, base-model]
timestamp: 2026-10-04T00:00:00Z
---

# DFM Mimir v1.5

Chosen as the base on 2026-10-04 instead of [Mimir v1](/research/mimir-v1.md). Pinned revision: `521b40b36a79918014544b970d4c2669ff1530eb` (last modified 2026-09-25).

## Facts (from the model card and config)

- Apache-2.0. BF16 **inference** weights only: no optimizer state, not a resumable training checkpoint.
- Final epoch-10 **EMA** weights at step 2,877,261, continuing the DFM10 epoch-9 checkpoint on the **DFM11** mixture (103.2B sampled tokens per epoch; a mixture size, not a lifetime budget). Final base learning rate 1e-5.
- 1,786,775,040 parameters in total, of which 981,468,672 are outside the input embedding and output head. The embedding and head are separate (262,144 vocabulary, hidden size 1536, untied).
- Architecture as in [HRM-Text](/research/hrm-text.md): 16 layers per H and L stack, 12 heads, H/L cycles 2/3, PrefixLM, context 4,096, Gemma4 tokenizer and chat template.
- Differs from v1 in `L_bp_cycles`: `[3, 3]` (v1: `[0, 3]`), meaning gradients flow through all L cycles in both H cycles. This raises activation memory when fine-tuning.
- Supported by transformers 5.13 as `hrm_text` (`HrmTextModel`, `HrmTextForCausalLM`); FlashAttention is rejected with `prefix_lm`, use `sdpa`.

## Training-data transparency

The repo ships `training_data_manifest.json` and `dfm11_sampling_policy.yaml`. The policy lists 193 dataset prefixes. It includes `tasksource__`, `flan__`, `flan_factual__`, `posttrain_natural_instructions__` and `sapient-synth-flan-*`, which matters for evaluation honesty; see [dataset-selection](/design/dataset-selection.md). The DFM10 base data it builds on is not public.
