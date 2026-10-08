---
type: Observation
title: 'O14: First pilot results: slot readout and letter baseline are tied'
description: One seed, validation n=6127. Slot and letter baseline differ by at most 0.004 in accuracy-level metrics; 96% of validation comes from sources Mimir saw.
date: '2026-10-09'
confidence: medium
tags: [pilot, e03, results]
timestamp: 2026-10-09T00:00:00Z
---

## Observation

Source: `runs/slot-v0/eval/*.json` and `runs/baseline-v0/eval/*.json` (committed in `a12853f`), made by `scripts/run_pilot.sh` with the default configs, one seed. Both evaluations are calibrated (one temperature per kind). The test split was not used. Compared with `python -m mimir_decide.compare runs/slot-v0 runs/baseline-v0 --file validation.json`.

| Validation, n = 6127 | Slot | Letter baseline |
|---|---|---|
| Accuracy | 0.831 | 0.835 |
| NLL | 0.455 | 0.456 |
| Brier | 0.202 | 0.201 |
| ECE | 0.018 | 0.016 |
| Noul (n=1079): accuracy / NLL / ECE | 0.905 / 0.265 / 0.029 | 0.922 / 0.262 / 0.045 |
| Choice (n=4030): accuracy / NLL / ECE | 0.852 / 0.416 / 0.019 | 0.852 / 0.418 / 0.019 |
| Score (n=1018): accuracy / NLL / ECE | 0.671 / 0.814 / 0.038 | 0.677 / 0.816 / 0.034 |
| Sources seen by Mimir (n=5872): accuracy / NLL | 0.833 / 0.438 | 0.835 / 0.440 |
| Not flagged, LocalLLaMA (n=255): accuracy / NLL | 0.804 / 0.851 | 0.835 / 0.846 |
| Held-out `massive/nb-NO` (n=600, all Choice): accuracy / NLL / ECE | 0.960 / 0.144 / 0.027 | 0.952 / 0.148 / 0.019 |

No decision was skipped for length (`skipped_too_long` is 0 for both).

## Interpretation

- **No measurable difference.** The gaps are 0.001 to 0.004 in NLL and Brier, and about 0.004 in accuracy. With one seed there is no estimate of run-to-run noise, so no direction can be claimed. The baseline is simpler, so on this evidence the slot readout has not earned its extra parts ([D05](/design/d05-slot-readout-heads.md), [D06](/design/d06-letter-logit-baseline.md)).
- **Both models are fine-tuned, not zero-shot.** The whole backbone of each (about 0.98B parameters) was trained for one epoch on the same mixture with the same loss. Calibration afterwards fits only 3 temperatures and does not change accuracy. The strong baseline therefore reflects Mimir's pretraining plus this fine-tuning.
- **Mimir already knows this kind of data.** 96% of validation comes from sources that Mimir's own training mix contains ([O07](/observations/o07-mimir-policy-includes-tasksource-and-flan.md), [O08](/observations/o08-audit-flag-rates.md)). The numbers show that the format conversion and a readout work. They do not show generalisation.
- **The held-out set is easy and probably familiar.** Accuracy 0.95 to 0.96 on MASSIVE nb-NO, which is probably covered by tasksource ([D15](/design/d15-held-out-task-massive-nb-no.md)).
- **Score is hardest** (accuracy about 0.67), as expected for ordinal ratings with neighbouring bins.
- **Missing reference points.** There is no zero-shot Mimir result with the letter prompt (how much did fine-tuning add?), no prior or majority baseline, and no second seed. The owner decided that one seed is enough, so the E03 decision rule needs a fixed margin instead of a seed spread.

## Reproduce

`python -m mimir_decide.compare runs/slot-v0 runs/baseline-v0 --file validation.json` and `--file heldout.json`.

## Related

* [E03](/experiments/e03-slot-vs-letter-pilot.md)
* [Q04](/questions/q04-slot-vs-letter-on-real-model.md)
* [Q01](/questions/q01-clean-generalisation-evidence.md)
