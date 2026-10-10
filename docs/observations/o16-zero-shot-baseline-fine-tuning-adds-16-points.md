---
type: Observation
title: 'O16: The untrained model reaches 0.669 accuracy; fine-tuning adds about 17 points'
description: Zero-shot Mimir v1.5 with the letter prompt scores 0.669 on validation against 0.835 for the fine-tuned baseline. The gain is largest on Score and smallest on the MASSIVE held-out task.
date: '2026-10-10'
confidence: medium
tags: [e10, zero-shot, pilot, results]
timestamp: 2026-10-10T00:00:00Z
---

## Observation

Source: `runs/letter-zeroshot/eval/*.json` (E10), `runs/baseline-v0` and `runs/slot-v0` (pilot), compared with `python -m mimir_decide.compare`. One seed. The zero-shot model is the untrained Mimir v1.5 with the letter prompt. The test split was not used.

| Validation, n = 6127 | Zero-shot, raw | Zero-shot, calibrated | Fine-tuned baseline | Slot model |
|---|---|---|---|---|
| Accuracy | 0.669 | 0.669 | 0.835 | 0.831 |
| NLL | 0.841 | 0.806 | 0.456 | 0.455 |
| Brier | 0.390 | 0.376 | 0.201 | 0.202 |
| ECE | 0.069 | 0.032 | 0.016 | 0.018 |
| Noul accuracy (n=1079) | 0.776 | 0.776 | 0.922 | 0.905 |
| Choice accuracy (n=4030) | 0.714 | 0.714 | 0.852 | 0.852 |
| Score accuracy (n=1018) | 0.377 | 0.377 | 0.677 | 0.671 |
| Seen by Mimir (n=5872) accuracy | 0.673 | 0.673 | 0.835 | 0.833 |
| Not flagged, LocalLLaMA (n=255) accuracy | 0.584 | 0.584 | 0.835 | 0.804 |
| Held-out `massive/nb-NO` (n=600) accuracy | 0.895 | 0.895 | 0.952 | 0.960 |
| Held-out NLL | 0.299 | 0.310 | 0.148 | 0.144 |
| Held-out ECE | 0.029 | 0.047 | 0.019 | 0.027 |

Fitted temperatures for the zero-shot model: noul 1.19, choice 1.29, score 1.84 (all above 1, so its raw probabilities were overconfident). No decision was skipped for length.

Training curves of the pilot (from the tracked `train_log.jsonl`, validation sample of 2,000 decisions): both runs took exactly 11,603 steps. Validation NLL fell from 0.591 (slot) and 0.597 (baseline) at step 1,000 to 0.461 and 0.462 at the end; accuracy rose from 0.766 and 0.769 to 0.815 and 0.827. Best validation NLL: 0.4592 (slot, step 11,000) and 0.4619 (baseline, step 11,603). Wall-clock time: 16.8 h (slot) and 15.8 h (baseline).

## Interpretation

- **E10 decision rule:** the gain over zero-shot is 0.835 - 0.669 = 16.6 points, which is above the threshold of 10 points. Fine-tuning adds most of the quality of the baseline. The earlier explanation that the baseline is strong mainly because Mimir already knows the data ([O14](/observations/o14-first-pilot-results-slot-and-baseline-tie.md)) is therefore only partly right: familiarity gives a solid start (0.669), but the fine-tuning does the larger part.
- **Where it matters:** Score gains the most (+30 points; the rubrics differ per task), then noul (+15) and choice (+14). On the held-out MASSIVE task the zero-shot model is already at 0.895 and fine-tuning adds about 6 points: Mimir mostly knows intent classification.
- **Do not over-read the not-flagged gain** (+25 points on LocalLLaMA): that data is synthetic and templated, so fine-tuning can learn its conventions.
- **Part of the gain is task format, not knowledge:** label names, rubric scales and question styles are learned during fine-tuning. A lower zero-shot number also reflects our own prompt, which is not Mimir's training format, so 0.669 is a lower bound for what the untrained model can do.
- **Most of the early gain comes quickly:** after 1,000 steps (about 32k decisions) the validation accuracy is already 0.77 on the 2,000-decision sample, and the next 10,600 steps add about 5 points. The two samples differ, so this is only a rough statement.
- **Calibration:** temperature scaling cut the zero-shot validation ECE from 0.069 to 0.032 but made the held-out result worse (ECE 0.029 to 0.047, NLL 0.299 to 0.310). One case, but it points the same way as the concern in [Q13](/questions/q13-calibration-transfer-across-domains.md): temperatures fitted on training-distribution data may not transfer.
- Slot and baseline remain tied; this result does not change that.

## Reproduce

`python -m mimir_decide.compare runs/letter-zeroshot runs/baseline-v0 runs/slot-v0 --file validation.json`, the same with `--file heldout.json`, and `--file validation_uncal.json` for the raw zero-shot numbers.

## Related

* [E10](/experiments/e10-zero-shot-letter-baseline.md)
* [Q14](/questions/q14-how-much-of-the-baseline-quality-is-fine-tuning.md)
* [O14](/observations/o14-first-pilot-results-slot-and-baseline-tie.md)
