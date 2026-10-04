---
type: Reference
title: Colleague recommendations (Mimir training lead)
description: What the colleague who has been training Mimir recommended for turning it into a decision model, and which parts we adopted.
tags: [recommendations, datasets, recipe]
timestamp: 2026-10-04T00:00:00Z
---

# Colleague recommendations

Received 2026-10-04 and pasted in by the project owner. A separate Mimir decision-model derivative is recommended rather than mixing these objectives into conversational SFT.

## Dataset candidates and our handling

| Candidate | Colleague's use | What we did |
|---|---|---|
| tasksource/tasksource-jev-typed-decisions (~2.5M decisions, 670 sources) | Strongest broad start; select sources, check licences | Adopted, default config (1.03M rows), permissive-licence filter |
| hotchpotch/bekko-system-one-dataset-v0 (6.59M cases) | Larger mixture; follow its training manifest, exclude quarantined, dedupe against tasksource | Adopted with manifest rules, per-subset licence audit, cross-dedup |
| LocalLLaMA/typed-decisions (1,200 train cases) | Multi-question over shared state; small and synthetic | Adopted (Apache-2.0) |
| AmazonScience/massive | Multilingual intent routing incl. Danish; humanise intent names | Adopted via the source archive, 8 candidate intents per utterance |
| nvidia/HelpSteer2 | Rubric scoring; ratings are not calibrated probabilities | Adopted as Score decisions, hard targets |
| ChaosNLI | Disagreement distributions; evaluation only | **Excluded**: licence is non-commercial, and it is built on SNLI/MNLI/αNLI dev sets |

tasksource and bekko overlap and must not be added together.

## In-house data (deferred)

DaLA acceptability probabilities, GEC pairs, Arena preferences (keep genuine ties), audits (keep/repair/reject/needs_verification, reviewed judgments only) and tool-choice examples. **Skipped for now** by the project owner; the converters do not exist yet.

## Recipe

1. Input is state, question and candidate descriptions. 2. Scoring head returns logits over supplied candidates, no answer strings. 3. Cross-entropy for hard labels, distributional cross-entropy for reliable soft targets. 4. Randomise option order and vary descriptions. 5. Fit calibration on a separate validation set; report accuracy, log loss, Brier and error when abstaining.

Proprietary RLCD need not be reproduced for a strong baseline. Soft labels or teacher probabilities are not automatically probabilities of correctness.

## Size and evaluation advice

Pilot of roughly 500K–1M carefully selected decisions (a proposal, not a demonstrated requirement). Hold out entire tasks and domains, and audit overlap with Mimir's earlier training. Broad Jev-level generalisation and reliable calibration still have to be shown, and removing autoregressive output does not remove the backbone's compute cost.

How this maps to implementation: [dataset-selection](/design/dataset-selection.md), [heads-and-baseline](/design/heads-and-baseline.md), [training-setup](/design/training-setup.md).
