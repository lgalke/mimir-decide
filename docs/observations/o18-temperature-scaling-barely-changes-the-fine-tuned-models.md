---
type: Observation
title: 'O18: Temperature scaling barely changes the fine-tuned models'
description: Raw and calibrated results of the slot model and the baseline differ by 0.001 to 0.009 in ECE and by at most 0.005 in NLL. Calibration does not help on the held-out task, and it worsens noul ECE.
date: '2026-10-11'
confidence: medium
tags: [e04, calibration, pilot, results]
timestamp: 2026-10-11T00:00:00Z
---

## Observation

Source: `runs/{slot-v0,baseline-v0}/eval/{validation,heldout}{,_uncal}.json`. The raw files use `--no_calibration`; the calibrated files apply the 3 fitted temperatures (E04). One seed. The re-evaluated calibrated files equal the files of the first pilot evaluation except for one added field, so the evaluation is reproducible. The server code did not yet contain the reliability tables, so none exist for these runs.

| Run, split, group | Raw acc | Raw conf | Raw ECE | Raw NLL | Calibrated conf | Calibrated ECE | Calibrated NLL |
|---|---|---|---|---|---|---|---|
| Slot, validation, all | 0.831 | 0.825 | 0.019 | 0.458 | 0.816 | 0.018 | 0.455 |
| Slot, validation, noul | 0.905 | 0.894 | 0.021 | 0.272 | 0.875 | 0.029 | 0.265 |
| Slot, validation, choice | 0.852 | 0.854 | 0.018 | 0.418 | 0.841 | 0.019 | 0.416 |
| Slot, validation, score | 0.671 | 0.635 | 0.047 | 0.814 | 0.655 | 0.038 | 0.814 |
| Slot, held-out nb-NO | 0.960 | 0.961 | 0.021 | 0.146 | 0.955 | 0.027 | 0.144 |
| Baseline, validation, all | 0.835 | 0.831 | 0.015 | 0.458 | 0.822 | 0.016 | 0.456 |
| Baseline, validation, noul | 0.922 | 0.894 | 0.036 | 0.267 | 0.877 | 0.045 | 0.262 |
| Baseline, validation, choice | 0.852 | 0.858 | 0.021 | 0.419 | 0.849 | 0.019 | 0.418 |
| Baseline, validation, score | 0.677 | 0.657 | 0.029 | 0.817 | 0.654 | 0.034 | 0.816 |
| Baseline, held-out nb-NO | 0.952 | 0.956 | 0.018 | 0.149 | 0.952 | 0.019 | 0.148 |

## Interpretation

- **The fine-tuned models are already calibrated without the post-hoc step.** Raw ECE is 0.015 to 0.019 overall. Temperature scaling changes NLL by at most 0.005 and ECE by 0.001 or less overall. Training with a proper scoring rule did the work.
- **By the E04 rule the temperatures do not transfer.** ECE does not fall on validation (slot 0.019 to 0.018, baseline 0.015 to 0.016) and it rises slightly on the held-out task for the slot model (0.021 to 0.027). The effect is small because the raw ECE is already small. This agrees with the zero-shot result ([O16](/observations/o16-zero-shot-baseline-fine-tuning-adds-16-points.md)), where calibration also worsened the held-out result.
- **Noul gets worse in ECE while NLL improves** (slot 0.021 to 0.029, baseline 0.036 to 0.045). The temperatures for noul are 1.27 and 1.23, which lower the confidence of a model that is already slightly underconfident (raw baseline noul: confidence 0.894, accuracy 0.922). Minimizing NLL punishes the few confidently wrong answers, and the best NLL temperature does not give the best ECE. This is why the calibrated models look slightly underconfident ([O17](/observations/o17-calibration-and-selective-risk-of-the-pilot-runs.md)). See [Q15](/questions/q15-is-nll-fitted-temperature-scaling-the-right-calibration.md).
- **Score** is the only kind where scaling helps ECE for the slot model (0.047 to 0.038), but for the baseline it does not (0.029 to 0.034).
- Slot and baseline are again equal.

## Reproduce

`python -m mimir_decide.compare runs/slot-v0 runs/baseline-v0 --file validation_uncal.json` and `--file validation.json` (same for `heldout`).

## Related

* [E04](/experiments/e04-calibration-effect.md)
* [Q13](/questions/q13-calibration-transfer-across-domains.md)
* [D19](/design/d19-calibration-protocol.md)
