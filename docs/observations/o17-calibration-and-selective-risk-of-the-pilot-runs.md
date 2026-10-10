---
type: Observation
title: 'O17: Calibration and selective risk of the pilot and zero-shot runs'
description: Fine-tuned models are well calibrated and slightly underconfident; their confidence ranks errors very well. The zero-shot model is overconfident, most on Score. Gaps in what is recorded are listed.
date: '2026-10-10'
confidence: medium
tags: [calibration, ece, brier, selective-risk, pilot, e04]
timestamp: 2026-10-10T00:00:00Z
---

## Observation

Source: `runs/*/eval/validation*.json` (validation, n = 6127, one seed). Each file stores accuracy, NLL, Brier, ECE, mean confidence, a selective-risk curve, score MAE and Spearman, per kind, source and language. `conf` is the mean top-1 confidence. `err@25/50%` is the error rate among the 25% / 50% most confident decisions, and `err@100%` is the overall error.

| Run | Kind | Acc | Conf | ECE | Brier | NLL | err@25% | err@50% | err@100% |
|---|---|---|---|---|---|---|---|---|---|
| Zero-shot, raw | all | 0.669 | 0.737 | 0.069 | 0.390 | 0.841 | 0.046 | 0.112 | 0.331 |
| Zero-shot, raw | score | 0.377 | 0.596 | 0.220 | 0.664 | 1.385 | 0.390 | 0.562 | 0.623 |
| Zero-shot, calibrated | all | 0.669 | 0.676 | 0.032 | 0.376 | 0.806 | 0.044 | 0.112 | 0.331 |
| Zero-shot, calibrated | noul | 0.776 | 0.761 | 0.035 | 0.223 | 0.479 | 0.022 | 0.078 | 0.224 |
| Zero-shot, calibrated | choice | 0.714 | 0.709 | 0.023 | 0.361 | 0.775 | 0.041 | 0.090 | 0.286 |
| Zero-shot, calibrated | score | 0.377 | 0.455 | 0.104 | 0.599 | 1.278 | 0.362 | 0.527 | 0.623 |
| Fine-tuned baseline | all | 0.835 | 0.822 | 0.016 | 0.201 | 0.456 | 0.001 | 0.014 | 0.165 |
| Fine-tuned baseline | noul | 0.922 | 0.877 | 0.045 | 0.074 | 0.262 | 0.000 | 0.000 | 0.078 |
| Fine-tuned baseline | choice | 0.852 | 0.849 | 0.019 | 0.193 | 0.418 | 0.001 | 0.010 | 0.148 |
| Fine-tuned baseline | score | 0.677 | 0.654 | 0.034 | 0.369 | 0.816 | 0.028 | 0.153 | 0.323 |
| Slot model | all | 0.831 | 0.816 | 0.018 | 0.202 | 0.455 | 0.001 | 0.012 | 0.169 |
| Slot model | noul | 0.905 | 0.875 | 0.029 | 0.075 | 0.265 | 0.000 | 0.000 | 0.095 |
| Slot model | choice | 0.852 | 0.841 | 0.019 | 0.193 | 0.416 | 0.002 | 0.010 | 0.148 |
| Slot model | score | 0.671 | 0.655 | 0.038 | 0.370 | 0.814 | 0.039 | 0.155 | 0.329 |

## Interpretation

- **The fine-tuned models are well calibrated on this validation set** (ECE 0.016 to 0.018) and slightly **underconfident**: mean confidence is 1 to 1.5 points below accuracy, and on noul 3 to 4.5 points below. Slot and baseline are tied on calibration as well; the only visible difference is noul ECE (slot 0.029, baseline 0.045; n = 1079, significance unknown).
- **The zero-shot model is overconfident** before calibration (confidence 0.737 against accuracy 0.669), most strongly on Score (0.596 against 0.377). Temperature scaling fixes the average (ECE 0.069 to 0.032) but Score stays overconfident (0.455 against 0.377).
- **Confidence ranks errors well in the fine-tuned models:** the error rate among the 50% most confident decisions is 1.2 to 1.4%, against 16.5 to 16.9% overall. For a decision model this is the useful property: abstaining on the less confident half removes almost all errors. Score is the exception (err@50% is 15% against 32% overall). The zero-shot model is much weaker here (11.2% against 33.1%). Caveat: validation is in-distribution, and 96% of it is familiar to Mimir.
- **Calibration here is calibration to the given labels** (hard labels for most sources, soft ones for some), not to truth ([Q07](/questions/q07-soft-label-semantics-vs-correctness.md)).

## What is not recorded yet

- **Raw (uncalibrated) results of the fine-tuned runs.** The pilot eval files are calibrated only, and `calibration.json` (ECE and NLL before and after on the calibration half) is ignored by git. The effect of temperature scaling on the fine-tuned models is therefore not in the repository. [E04](/experiments/e04-calibration-effect.md) produces it (`evaluate --no_calibration --tag uncal`; the checkpoints are on the server).
- **Reliability tables.** Only the scalar ECE was stored. The code now also writes a `reliability` list (15 bins: count, mean confidence, accuracy) for `all` and each kind. The pilot files predate this; re-evaluating the pilot checkpoints adds it.
- **Definition limits.** ECE uses the top-1 confidence, 15 equal-width bins, and counts a decision as correct if the chosen option has maximal target mass; there is no class-wise ECE. Brier is the sum over options of the squared error (range 0 to 2), so it depends on the number of options and should only be compared within the same task mix.
- **No confidence intervals.** Differences of a few thousandths between runs are within any plausible noise with one seed.
- **Held-out calibration** of the fine-tuned runs: the held-out files are calibrated only ([Q13](/questions/q13-calibration-transfer-across-domains.md)).

## Reproduce

`python -m mimir_decide.compare runs/letter-zeroshot runs/baseline-v0 runs/slot-v0 --file validation.json` (now shows `conf` and `err@50%cov`), and `--file validation_uncal.json` for the raw zero-shot run.

## Related

* [O14](/observations/o14-first-pilot-results-slot-and-baseline-tie.md)
* [O16](/observations/o16-zero-shot-baseline-fine-tuning-adds-16-points.md)
* [E04](/experiments/e04-calibration-effect.md)
