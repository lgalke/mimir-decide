---
type: Observation
title: 'O23: The letter probe (head only, frozen backbone) recovers about a fifth of the fine-tuning gain'
description: Training only the output head, started from the language-model head, lifts validation accuracy from 0.669 to 0.703, mostly on score questions, and does not help on the Typed Decisions benchmark. The fine-tuning gain comes from the backbone.
date: '2026-10-13'
confidence: medium
tags: [e13, probe, frozen-backbone, results]
timestamp: 2026-10-13T00:00:00Z
---

## Observation

Source: `runs/letter-probe/` (E13: letter baseline, frozen backbone and embeddings, `train_letter_head=true`, head learning rate 1e-3 fixed in advance, the pilot's data and settings, one seed), compared with `runs/letter-zeroshot`, `baseline-v0`, `baseline-lmhead` and `slot-v0`. The head is a trainable copy of the 26 letter rows of the language-model head, so step 0 is the zero-shot model. The slot probe has not been run yet.

| Calibrated, validation (n=6127) | Zero-shot | **Letter probe** | Frozen-head baseline | Trainable-head baseline |
|---|---|---|---|---|
| Accuracy | 0.669 | **0.703** | 0.835 | 0.833 |
| NLL | 0.806 | **0.766** | 0.456 | 0.454 |
| Brier / ECE | 0.376 / 0.032 | **0.354 / 0.013** | 0.201 / 0.016 | 0.200 / 0.017 |
| Noul accuracy (n=1079) | 0.776 | **0.792** | 0.922 | 0.908 |
| Choice accuracy (n=4030) | 0.714 | **0.713** | 0.852 | 0.852 |
| Score accuracy (n=1018) | 0.377 | **0.568** | 0.677 | 0.680 |
| Seen by Mimir (n=5872) accuracy | 0.673 | **0.707** | 0.835 | 0.833 |
| Not flagged, LocalLLaMA (n=255) accuracy | 0.584 | **0.604** | 0.835 | 0.843 |
| Held-out nb-NO (n=600) accuracy / NLL | 0.895 / 0.310 | **0.905 / 0.293** | 0.952 / 0.148 | 0.963 / 0.136 |

**Typed Decisions benchmark** (calibrated, 2,000 decisions): the probe reaches accuracy 0.611, KL from gold 0.301, Brier 0.164 (untrained model: 0.607, 0.295, 0.166; fine-tuned models 0.78-0.79, 0.086-0.087, 0.048). Per kind: noul 0.622, choice 0.572, score 0.631 (untrained: 0.633, 0.563, 0.619). Per workflow the probe is within 1.2 points of the untrained model on all four.

**Training:** 11,603 steps, 3.6 h, 28.8 examples/s (4.5 times faster than a full run), peak memory 9.04 GB. Validation NLL on the 2,000-decision sample: 0.811 at step 1,000, 0.761 at 4,000, 0.730 at 10,000; best 0.7264. Fitted temperatures are close to 1 (1.06, 1.06, 1.08).

## Interpretation

- **E13 rule (fixed beforehand):** the share of the gap that the head recovers is R = (0.703 - 0.669) / (0.835 - 0.669) = **0.20** against the frozen-head baseline (0.21 against the trainable-head baseline). That is below 0.25, so **the fine-tuning gain comes mostly from the backbone, not from the readout.**
- **What the head can learn:** nearly all of its gain is on score questions (0.377 to 0.568, +19 points) where the rubric scales and their direction are conventions that a readout can adapt. It learns nothing on choice questions (0.714 to 0.713) and little on noul (+1.6 points).
- **On the benchmark the probe is no better than the untrained model** (+0.4 points accuracy, KL 0.301 against 0.295), although the training mixture contained the benchmark's train split. A head over frozen features cannot learn the four workflows; the backbone has to change. This agrees with the large gain of full fine-tuning on the same data ([O22](/observations/o22-typed-decisions-benchmark-results.md)).
- **The probe is the best calibrated model on our validation data** (ECE 0.013) but its confidence ranks errors poorly (error among the most confident half 10.3% against 1.1-1.4% for the fine-tuned models).
- **Limits:** one seed; the learning rate 1e-3 and 11,603 steps were fixed in advance and not tuned; the curve is noisy at the end (NLL 0.76 at step 7,000, 0.73 at step 10,000). A better-tuned head might gain somewhat more, but its capacity is 26 x 1,536 numbers over fixed features.
- **Open:** the slot probe, which decides the second E13 question (does the slot readout extract more from the same frozen features: the rule is a difference of at least 2 points of accuracy against this probe).

## Reproduce

`python -m mimir_decide.compare runs/letter-zeroshot runs/letter-probe runs/baseline-v0 runs/baseline-lmhead runs/slot-v0 --file validation.json`; benchmark numbers from `runs/letter-probe/eval/typed_decisions_test.json`.

## Related

* [E13](/experiments/e13-head-only-baselines.md)
* [O16](/observations/o16-zero-shot-baseline-fine-tuning-adds-16-points.md)
* [Q14](/questions/q14-how-much-of-the-baseline-quality-is-fine-tuning.md)
