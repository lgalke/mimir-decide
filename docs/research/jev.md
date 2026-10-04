---
type: Product
title: Jev (TypeSafe AI)
description: First commercial decision model; API returning calibrated Choice/Score/Noul outputs at 70-500 ms.
resource: https://jevai.net/articles/what-is-system-one-jev/
tags: [decision-model, jev, typesafe, api]
timestamp: 2026-10-03T00:00:00Z
---

# Jev

Early access released 2026-09-15 by TypeSafe AI. Described as "neither small nor an LLM": "unstructured state in, typed probabilistic decisions out". It is a [decision model](/research/decision-models.md).

## Interface

A request has a **state** (strings, arrays, name-value pairs) and several **questions** (Noul / Choice / Score). Questions run in parallel against one state.

## Claimed numbers (vendor-reported)

- Price: $0.042 per 1M input tokens, output free (GPT-5 Nano: $0.05).
- Latency: 70-500 ms end to end; 114 ms and 236-276 ms in two separate reports.
- 0% structured-output errors.
- Vendor benchmark: ~$0.000081 per call, 444.6x cheaper and 193.6x faster than comparable LLMs.
- LangChain validation: 500/500 agreement with a human oracle; variance 433-913x lower than LLM judges.
- Calibration: ECE 0.246 on a proprietary eval set (third-party pentest paper; see [calibration](/research/calibration-training.md)). Not impressive in absolute terms.

## Training and architecture

- Training: "Reinforcement Learning for Calibrated Decisions" (RLCD), details undisclosed. See [calibration-training](/research/calibration-training.md).
- Architecture: a "parallel sampler" that emits all outputs in one query. Model size, backbone, and head design are **not public**.

## Known weaknesses

Numbers, dates, adversarial content (Jev 1.13 docs). Bias and black-box concerns raised by Willison, with no mitigations disclosed.

Related: [Laya](/research/laya.md).
