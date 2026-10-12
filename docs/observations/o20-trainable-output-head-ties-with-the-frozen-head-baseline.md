---
type: Observation
title: 'O20: The baseline with a trainable output head ties with the frozen-head baseline and the slot model'
description: Training the output head of the letter baseline changes validation accuracy by 0.2 points and NLL by 0.002. All three trained models are indistinguishable on every in-distribution measure.
date: '2026-10-12'
confidence: medium
tags: [e11, baseline, pilot, results]
timestamp: 2026-10-12T00:00:00Z
---

## Observation

Source: `runs/baseline-lmhead/` (E11: letter baseline, `train_lm_head=true`, the pilot's data and settings, one seed), compared with `runs/baseline-v0` (frozen head) and `runs/slot-v0` with `python -m mimir_decide.compare`. The files for `baseline-lmhead` are the first to contain reliability tables. The test split was not used.

| Calibrated | Frozen-head baseline | **Trainable-head baseline** | Slot model |
|---|---|---|---|
| Validation accuracy (n=6127) | 0.835 | **0.833** | 0.831 |
| Validation NLL | 0.456 | **0.454** | 0.455 |
| Validation Brier / ECE | 0.201 / 0.016 | **0.200 / 0.017** | 0.202 / 0.018 |
| Error among the 50% most confident | 0.014 | **0.011** | 0.012 |
| Noul accuracy / NLL (n=1079) | 0.922 / 0.262 | **0.908 / 0.264** | 0.905 / 0.265 |
| Choice accuracy / NLL (n=4030) | 0.852 / 0.418 | **0.852 / 0.411** | 0.852 / 0.416 |
| Score accuracy / NLL (n=1018) | 0.677 / 0.816 | **0.680 / 0.824** | 0.671 / 0.814 |
| Seen by Mimir (n=5872) accuracy | 0.835 | **0.833** | 0.833 |
| Not flagged, LocalLLaMA (n=255) accuracy | 0.835 | **0.843** | 0.804 |
| Held-out `massive/nb-NO` (n=600) accuracy / NLL / ECE | 0.952 / 0.148 / 0.019 | **0.963 / 0.136 / 0.011** | 0.960 / 0.144 / 0.027 |

Further results for the trainable-head baseline:

- **Raw against calibrated** (validation): ECE 0.014 raw, 0.017 calibrated; NLL 0.456 raw, 0.454 calibrated. Per kind, calibration lowers ECE on choice (0.017 to 0.013) and raises it on noul (0.021 to 0.035) and score (0.031 to 0.041). On the held-out task: ECE 0.012 raw, 0.011 calibrated. Temperatures: noul 1.29, choice 1.09, score 1.03. The pattern matches [O18](/observations/o18-temperature-scaling-barely-changes-the-fine-tuned-models.md).
- **Option order** (3 random orders, validation accuracy): given order 0.8332, mean over the random orders 0.8335, drop -0.0003, spread 0.0023. Frozen-head baseline: drop +0.0017; slot model: drop +0.0003. No model depends on the option order ([O19](/observations/o19-option-order-does-not-matter-for-either-model.md)).
- **Reliability table:** half of the validation decisions (3,093 of 6,127) have a confidence above 0.93; their mean confidence is 0.988 and their accuracy 0.989.
- **Training** (all three runs: 11,603 steps): trainable-head baseline 16.1 h, 6.41 examples/s, peak memory 37.99 GB, which is 4.4 GB more than the frozen-head baseline (33.55 GB). Validation NLL on the 2,000-decision sample at step 1,000: 0.640 (trainable head), 0.597 (frozen head), 0.591 (slot); at the end: 0.4612, 0.4619 and 0.4610. The trainable head starts slower and ends at the same place.

## Interpretation

- **E11 rule (fixed beforehand):** a validation NLL difference below 0.01 is a tie, and a tie goes to the simpler model. The difference to the frozen-head baseline is 0.002 and to the slot model 0.001, so it is a **tie**, and the frozen-head baseline is preferred (4.4 GB less memory, no extra parameters). The 0.01 margin is a proposal of the agent that the owner has not confirmed.
- **The asymmetry was not the reason for the tie.** The pilot compared a slot model with a trained head against a baseline with a frozen head. With the head trained as well, nothing changes. This answers the fairness concern behind E11.
- **Small differences, all within noise:** the trainable head is 0.007 better in choice NLL and 1.1 points better on the held-out task (n = 600, about one standard error), and 1.4 points worse in noul accuracy (n = 1079). With one seed none of these can be called a difference.
- **Calibration is a property of the training, not of the readout:** all three models are well calibrated raw, and the post-hoc temperatures help little ([Q15](/questions/q15-is-nll-fitted-temperature-scaling-the-right-calibration.md)).
- The three trained models are still indistinguishable on in-distribution measures. What can separate them is outside the training distribution: the extra tasks of [E12](/experiments/e12-unseen-label-sets.md), and the probes of [E13](/experiments/e13-head-only-baselines.md).

## Reproduce

`python -m mimir_decide.compare runs/letter-zeroshot runs/baseline-v0 runs/baseline-lmhead runs/slot-v0 --file validation.json` (and `heldout.json`).

## Related

* [E11](/experiments/e11-stronger-baseline-trainable-lm-head.md)
* [O14](/observations/o14-first-pilot-results-slot-and-baseline-tie.md)
* [Q04](/questions/q04-slot-vs-letter-on-real-model.md)
