---
type: Observation
title: 'O22: On the Typed Decisions test split the fine-tuned models reach accuracy 0.788 and KL 0.086, level with fine-tuned LoRA models'
description: First benchmark results of our four models (full test split, 2,000 decisions). The three fine-tuned models are tied; the untrained model reaches 0.607.
date: '2026-10-12'
confidence: medium
tags: [e14, benchmark, typed-decisions, results]
timestamp: 2026-10-12T00:00:00Z
---

## Observation

Source: `runs/<run>/eval/typed_decisions_test*.json` (E14), scored once per model with `mimir_decide.typed_decisions` (definitions in [O21](/observations/o21-benchmark-scorer-matches-the-published-uniform-row-only-for-kl-and-brier.md)). One question per request. The fine-tuned models include the benchmark's train split in a broad training mixture; the zero-shot model is untrained.

| Calibrated | Accuracy | KL from gold | Brier | ECE (hard) | Mean confidence | NLL |
|---|---|---|---|---|---|---|
| Untrained Mimir, letter prompt (zero-shot) | 0.607 | 0.295 | 0.166 | 0.095 | 0.590 | 1.063 |
| Untrained Mimir, raw | 0.607 | 0.391 | 0.195 | 0.076 | 0.671 | 1.159 |
| Slot model (`slot-v0`) | 0.788 | 0.086 | 0.048 | 0.159 | 0.630 | 0.854 |
| Frozen-head baseline (`baseline-v0`) | 0.788 | 0.087 | 0.048 | 0.158 | 0.630 | 0.854 |
| Trainable-head baseline (`baseline-lmhead`) | 0.779 | 0.086 | 0.048 | 0.161 | 0.618 | 0.854 |

Per kind (slot model): noul 0.875, choice 0.748, score 0.754. Per workflow (slot model): agent traces 0.726, customer service 0.834, invoices 0.820, security incidents 0.774. Gain of fine-tuning over zero-shot by workflow (slot model, accuracy points): agent traces +24.4 (0.482 to 0.726), invoices +18.6, security +19.6, customer service +10.2.

**Paired comparisons** (differences over the same 2,000 decisions, 95% bootstrap interval over the 400 cases):

| A - B | Accuracy (points) | KL |
|---|---|---|
| slot - frozen-head baseline | +0.00 [-1.35, +1.35] | -0.0006 [-0.0036, +0.0023] |
| trainable-head baseline - frozen-head baseline | -0.95 [-2.10, +0.25] | -0.0005 [-0.0035, +0.0024] |
| slot - trainable-head baseline | +0.95 [-0.45, +2.35] | -0.0001 [-0.0030, +0.0028] |

**Questions:** lowest accuracy of the slot model: `agent_trace_observability/urgency` 0.600, `security_incidents/urgency` 0.650, `customer_service/action` 0.670, `agent_trace_observability/action` 0.680, `invoice_processing/disposition` 0.690. Highest: `customer_service/category` 0.950, `invoice_processing/matches_order` 0.940, `invoice_processing/duplicate` 0.910.

**Position among the README's published rows** (published rows are mostly self-reported, were scored with the submitters' own accuracy and ECE definitions, and send all five questions at once; KL and Brier use the same definitions as ours):

| Group | Row | Accuracy | KL | Brier |
|---|---|---|---|---|
| Zero-shot | meraGPT Decider 1 | 0.768 | 0.096 | 0.052 |
| Zero-shot | OpenDecider-small | 0.671 | 0.211 | 0.117 |
| Zero-shot | **Untrained Mimir (ours)** | **0.607** | **0.295** | **0.166** |
| Zero-shot | Bongard-mini | 0.594 | 0.256 | 0.132 |
| Zero-shot | Jeff-Gemma4-E2B | 0.561 | 0.403 | 0.219 |
| Zero-shot | Prior (ignores the input) | 0.470 | 0.347 | 0.189 |
| Trained on `train` | Phocinae-Largha-150M (specialist) | 0.906 | 0.0548 | 0.0248 |
| Trained on `train` | OpenDecider-large-td (LoRA) | 0.801 | 0.081 | 0.044 |
| Trained on `train` | OpenDecider-small-td (LoRA) | 0.792 | 0.080 | 0.043 |
| Trained on `train` | **Slot / frozen-head baseline (ours)** | **0.788** | **0.086-0.087** | **0.048** |
| Trained on `train` | OpenDecider-medium-td (LoRA) | 0.788 | 0.081 | 0.044 |
| Trained on `train` | soft-decider-421m | 0.774 | - | - |
| Trained on `train` | Bekko System One 400M | 0.668 | 0.214 | 0.113 |
| Trained on `train` | ModernBERT-base specialist | 0.646 | 0.223 | 0.119 |

## Interpretation

- **The three fine-tuned models are tied** again, now with paired intervals: the slot model and the frozen-head baseline differ by 0.00 points (interval about +-1.35), and no KL difference is above 0.004. The slot design brings nothing here either.
- **Level with fine-tuned LoRA models in accuracy, slightly behind in KL and Brier** (0.086 against 0.080-0.081, 0.048 against 0.043-0.044), and well behind the specialist Phocinae (0.906, a 150M encoder fitted to these four workflows only). Our model is a general 1B model trained on a mixture in which this benchmark's train split is a small part, and it answers every question separately.
- **The untrained model is mid-field among the zero-shot rows** in accuracy (0.607) and better than most in KL (0.295). Fine-tuning adds 18 accuracy points and cuts KL by a factor of 3.4.
- **Scores above the 0.735 ceiling** (teacher self-agreement) mean, according to the README, that a model learns the teacher's quirks. All fine-tuned models, ours included, are above it. On `agent_trace_observability/urgency`, whose README ceiling is 0.560, the slot model reaches 0.600.
- **Hard-label ECE is not informative here** (0.15-0.16 for the fine-tuned models while they are calibrated on our own validation). The gold is the mean of three teacher samples, so even a perfect prediction of the gold distribution has top probability around 0.6 and looks 'underconfident' against hard agreement (mean confidence 0.63, accuracy 0.79). The README says the same: the Prior has the best ECE, and KL and Brier are the columns to read.
- **Our temperatures help the zero-shot model a lot and the fine-tuned models not at all:** zero-shot KL 0.391 to 0.295 and Brier 0.195 to 0.166; fine-tuned KL 0.086-0.090 to 0.086-0.087, Brier unchanged. The temperatures were fitted on our general validation data, not on this benchmark.

## Reproduce

`python -m mimir_decide.typed_decisions --run_dir runs/<run> --final` (add `--checkpoint zero-shot` for the untrained model); the paired intervals come from the saved `*_predictions.json` files, bootstrapped over the 400 cases.

## Related

* [E14](/experiments/e14-typed-decisions-benchmark.md)
* [O20](/observations/o20-trainable-output-head-ties-with-the-frozen-head-baseline.md)
* [D28](/design/d28-typed-decisions-scorer-and-test-use.md)
