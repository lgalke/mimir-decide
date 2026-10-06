---
type: Reference
title: RLCD and calibration training
description: "Training objectives for calibrated decisions: RLCD (Jev), RLCR, Brier-score rewards, temperature scaling, RLHV."
tags: [calibration, rl, brier, training]
timestamp: 2026-10-03T00:00:00Z
---

# RLCD and calibration training

| Method                                      | Optimises                                             | Calibration                           |
| ------------------------------------------- | ----------------------------------------------------- | ------------------------------------- |
| RLHF                                        | human preference                                      | no guarantee, overconfident           |
| RLAIF                                       | AI-generated preference                               | inherits labeler miscalibration       |
| RLVR                                        | verifiable correctness (binary)                       | tends to hurt calibration             |
| **RLCD** (TypeSafe/[Jev](/research/jev.md)) | proper scoring rule on decision tasks                 | by construction (details undisclosed) |
| **RLCR** (arXiv 2507.16806)                 | binary correctness + Brier score on stated confidence | ECE 0.37 to 0.03, accuracy steady     |
| RLHV (proposed in pentest paper)            | verdicts validated by deterministic checkers          | closed-loop self-improvement          |

## Key facts

- A proper scoring rule such as Brier is maximised when output = true probability. RLCR proves its reward is maximised by answering the most probable answer with confidence equal to the true success probability.
- For a non-generative model that outputs probabilities directly, you do not need RL at all to apply this: **train with a proper scoring loss (cross-entropy / Brier) on labelled decisions**, then apply temperature scaling on held-out data. RL is needed when labels are only available via a verifier or when sampling is involved. (My inference; Jev's RLCD internals are not public.)
- Post-hoc temperature scaling is cheap and effective (Laya 0.466 to 0.081 ECE, see [Laya](/research/laya.md)).
- Metric to track: ECE, Brier, reliability diagrams, plus accuracy per question type.

Used in the [plan](/design/mimir-to-decision-model-plan.md).
