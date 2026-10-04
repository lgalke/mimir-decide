---
type: Architecture
title: HRM-Text
description: Hierarchical recurrent language model with slow H-module and fast L-module cycles; basis of Mimir.
resource: https://arxiv.org/html/2605.20613v1
tags: [hrm, recurrent, architecture]
timestamp: 2026-10-03T00:00:00Z
---

# HRM-Text

- Two coupled recurrent modules: **H** (slow, strategic) and **L** (fast, local refinement). Standard pass: 2 H-cycles x 3 L-updates.
- Stabilisation: MagicNorm (PreNorm blocks plus final norm); shallow-to-deeper truncated credit assignment (last 2 steps growing to 5).
- Objective: response-only loss, PrefixLM mask (bidirectional on instruction, causal on response).
- Original: 40B unique tokens, 16 H100 for 46 h (~$1,500), Adam-atan2, constant LR 2.2e-4. 1B model: MMLU 60.7, GSM8K 84.5, MATH 56.2, using ~96-432x less compute than 2-7B peers.
- Adaptive computation (ACT) discussed as future work, not implemented.

## Relevance

Recurrent depth gives iterative refinement of a hidden state before a final readout, a natural fit for reading out a decision vector. ACT would map to early exit on easy decisions. See [plan](/design/mimir-to-decision-model-plan.md) and [Mimir](/research/mimir-v1.md).
