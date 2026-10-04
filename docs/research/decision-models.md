---
type: Concept
title: Decision Models (System One models)
description: Non-generative models that read text and return typed, calibrated probabilistic decisions instead of tokens.
tags: [decision-model, system-one, calibration]
timestamp: 2026-10-03T00:00:00Z
---

# Decision Models (System One models)

"System One" borrows from Kahneman's dual-process theory: fast, bounded, pattern-matching judgment, as opposed to slow deliberative reasoning ("System Two", i.e. chain-of-thought LLMs). The term was popularised by TypeSafe AI with [Jev](/research/jev.md) (early access 2026-09-15); Simon Willison suggested "decision models" is the better name.

## Defining properties

- **Input:** unstructured text/state (documents, records, agent traces), plus one or more *questions*.
- **Output:** floats, not text. Three primitives (shared by [Jev](/research/jev.md) and [Laya](/research/laya.md)):
  - **Noul**: yes/no proposition, probability in [0,1] (Bernoulli).
  - **Choice**: pick from a finite option set, returns a distribution plus confidence.
  - **Score**: rating on an ordered rubric/numeric range, probability-weighted.
- **Schema-constrained:** only schema-defined values can come out, so structured-output errors are 0% by construction (claimed 0% for Jev).
- **Parallel:** all questions/options are evaluated in a single pass, independently. No autoregressive decoding, no cascading errors between questions, latency roughly constant in the number of questions.
- **Calibrated:** stated confidence should match empirical accuracy. See [calibration training](/research/calibration-training.md).

## Where it fits

Classification, scoring, guardrails, routing, LLM-as-judge replacement, batch evaluation, agent-harness decision points (finding adjudication, pruning, confirmation loops; see the pentest paper in [sources](/research/sources.md)).

## Where it does not fit

Long-form generation, mathematical derivation, open-ended exploration, chain-of-thought. Vendor docs also list weaknesses with numbers, dates and adversarial content.

## Relation to older ideas (my interpretation, not claimed by vendors)

Functionally close to a calibrated multi-head text classifier / reward model / cross-encoder, but with questions specified at *inference time* in natural language rather than fixed label heads. This is the key capability to reproduce: open-vocabulary questions answered by a numeric head.
