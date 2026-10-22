---
type: Observation
title: 'O28: Check of the Mimir basics: PrefixLM, tokenizer, chat template and learning-rate scaling'
description: PrefixLM attention, tokenizer and chat template are used as in Mimir's training. Mimir's own final training scaled the learning rate per stack (H by 1/2, L by 1/6), which our runs so far did not. The slot and scalar prompts are plain text, not the chat format.
date: '2026-10-22'
confidence: high
tags: [prefixlm, tokenizer, chat-template, learning-rate, basics]
timestamp: 2026-10-22T00:00:00Z
---

## Observation

Sources: the Mimir v1.5 model card (revision `521b40b3`), the HRM-Text repository (`dataset_new.py`, `scripts/tokenize_chat_template.py`, `evaluation/chat_templates/`), the installed `transformers` `hrm_text` code, and new tests (`tests/test_mimir_basics.py`, run on the real tokenizer and on a tiny random backbone).

| Basic | What Mimir does | What we do | Status |
|---|---|---|---|
| PrefixLM | Each example is an instruction (prefix, bidirectional) and a response (causal); the last prefix token predicts the first response token (`dataset_new.py`) | Every real token has `token_type_ids = 1`, so the whole prompt is one bidirectional block; right padding with `attention_mask = 0`; readout at the last prompt token (letter) or at option markers/last option token (slot, scalar) | matches; tested: a changed last token changes earlier states with type 1 and does not with type 0; padding does not change real tokens |
| Prefix content | The prefix is the chat template rendered **with the generation prompt**: `<bos><|turn>user\n...<turn|>\n<|turn>model\n`; the response is the rest (`tokenize_chat_template.py`: `add_generation_prompt=True`, `enable_thinking=False`, `add_special_tokens=False`) | The letter prompt is exactly this string (test against `apply_chat_template`); the template output is the same with thinking off and default | matches |
| Tokenizer | Gemma 4 tokenizer, 262,144 tokens, `<bos>` 2, `<pad>` 0, EOS `<turn|>` 106; the default call adds `<bos>` | Loaded from the pinned Mimir revision; we call it with `add_special_tokens=False` and add `<bos>` once ourselves; `<unused0>` = 6 (slot marker); letters A to Z are single tokens | matches |
| Attention kernel | FlashAttention PrefixLM kernels | `sdpa` with the transformers PrefixLM mask (FlashAttention is rejected with `prefix_lm`) | same mask semantics, tested |
| Gradient through cycles | "BP8": gradients through all 8 module executions (H2/L3) | `L_bp_cycles = [3, 3]` kept from the checkpoint | matches |
| Precision | BF16 compute, FP32 parameters and optimizer state | bf16 autocast, fp32 master weights | matches |
| **Learning rate** | Final base rate 1e-5 for embeddings/head with **automatic module scaling: H 5e-6 (base/2), L about 1.6667e-6 (base/6)** (model card) | One rate 1e-5 for the whole backbone in all runs so far (`lr_cycle_dividers` off) | **differs**: our L stack learns 6 times and our H stack 2 times faster than Mimir's last training steps |
| Prompt format of the readouts | Chat format (Gemma-native turns) | Letter baseline: chat format. Slot and scalar: plain `State:/Question:/Option(s):` text without turn tokens | **differs** for slot and scalar |

## Interpretation

- **No error found in the PrefixLM mask, the tokenizer or the chat template** for the letter baseline, which is running now. The tests are in the suite.
- **Learning rate:** the flag `lr_cycle_dividers` ([D30](/design/d30-galke-poech-lr-dividers.md)) reproduces exactly the scaling that the model card describes (L divided by 6, H by 2 at base 1e-5). The current 1M run (letter baseline) and all earlier runs have it off. The earlier results (0.835 validation accuracy) show the setting works; whether the Mimir scaling is better for this fine-tuning is untested. For further runs the flag with `lr_backbone: 1.0e-5` gives the Mimir step sizes.
- **Prompt format:** for fully fine-tuned slot and scalar models the plain-text prompt is learned; for frozen probes it is out of the training format, which may contribute to the weak slot probe ([O24](/observations/o24-slot-probe-extracts-little-from-a-frozen-backbone.md)); this was not tested. A chat-format variant (the same text inside the user turn, the readout at the generation prompt) would be native to the model; not implemented.
- **Not checked:** the optimizer family of Mimir v1.5 (the original HRM-Text used Adam-atan2; we use AdamW with betas 0.9 and 0.95), and what the model emits first after `<|turn>model\n` with thinking off.

## Reproduce

`python -m pytest tests/test_mimir_basics.py -q -m "network or not network"`; model card and scripts as listed above.

## Related

* [D30](/design/d30-galke-poech-lr-dividers.md)
* [D09](/design/d09-keep-checkpoint-l-bp-cycles.md)
* [D06](/design/d06-letter-logit-baseline.md)
* [E16](/experiments/e16-mixture-v1-training.md)
