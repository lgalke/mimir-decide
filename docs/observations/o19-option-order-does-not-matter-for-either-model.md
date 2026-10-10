---
type: Observation
title: 'O19: Neither model is sensitive to option order'
description: With 3 random option orders, validation accuracy changes by 0.03 points (slot) and 0.17 points (baseline). The pre-set rule does not support the slot design's order-invariance advantage.
date: '2026-10-11'
confidence: medium
tags: [e05, option-order, pilot, results]
timestamp: 2026-10-11T00:00:00Z
---

## Observation

Source: `runs/{slot-v0,baseline-v0}/eval/validation.json` (given order) and `validation_order{1,2,3}.json` (`--order_seed 1..3`: options in a seeded random order, score options in given or reversed order; logits mapped back). Calibrated with the temperatures of the given order. One training seed. `drop` is the given-order accuracy minus the mean accuracy over the 3 random orders.

| Model | Kind | Given order | Random orders | Mean | Drop | Spread |
|---|---|---|---|---|---|---|
| Slot | all | 0.831 | 0.832, 0.831, 0.830 | 0.831 | +0.0003 | 0.0018 |
| Slot | noul | 0.905 | 0.905, 0.907, 0.904 | 0.905 | -0.0009 | 0.0037 |
| Slot | choice | 0.852 | 0.852, 0.850, 0.851 | 0.851 | +0.0013 | 0.0017 |
| Slot | score | 0.671 | 0.675, 0.673, 0.672 | 0.673 | -0.0023 | 0.0029 |
| Baseline | all | 0.835 | 0.833, 0.833, 0.834 | 0.833 | +0.0017 | 0.0016 |
| Baseline | noul | 0.922 | 0.920, 0.921, 0.923 | 0.922 | +0.0006 | 0.0028 |
| Baseline | choice | 0.852 | 0.850, 0.849, 0.851 | 0.850 | +0.0014 | 0.0022 |
| Baseline | score | 0.677 | 0.674, 0.673, 0.672 | 0.673 | +0.0039 | 0.0020 |

NLL stays within 0.003 of the given-order value in every row (for example slot overall 0.455 to 0.455, 0.455, 0.457; baseline 0.456 to 0.457, 0.457, 0.458).

## Interpretation

- **Rule outcome (fixed before the results were read):** the baseline's drop (+0.17 points) minus the slot model's drop (+0.03 points) is 0.14 points, below the 1-point margin. The invariance argument of [D05](/design/d05-slot-readout-heads.md) is **not supported**; both models are equally robust to option order, and the simpler baseline is preferred on this criterion.
- Both models trained with shuffled option order, so both learned to bind answers to option content. The toy experiment, in which the letter baseline learned shuffled options slowly ([O09](/observations/o09-toy-slot-vs-letter.md)), did not carry over to the real model.
- All spreads are at most 0.4 points, within what random option permutations of the same model can cause.
- **Not computed:** the agreement of the predicted option between orders. The eval files hold only aggregate metrics, not per-decision predictions. The rule did not depend on it.

## Reproduce

`python -m mimir_decide.evaluate --run_dir runs/<run> --data_dir $DATA --split validation --order_seed 1` (and 2, 3), then compare the `acc` values.

## Related

* [E05](/experiments/e05-option-order-robustness.md)
* [Q12](/questions/q12-option-position-bias-at-evaluation.md)
* [O14](/observations/o14-first-pilot-results-slot-and-baseline-tie.md)
