---
type: Observation
title: 'O30: The scalar per-option head ties with the letter baseline and costs three times the training time'
description: On the v0 mixture at 2048 context the scalar head matches baseline-v0 on validation, held-out and unseen label sets (E12 gain +8.4 against +8.9, wins on 10 of 32 sources). By the E17 rule it is a tie. Typed Decisions KL is lowest of all models (0.078), descriptive only.
date: '2026-10-25'
confidence: medium
tags: [e17, scalar, heads, results]
timestamp: 2026-10-25T00:00:00Z
---

## Observation

Run `runs/scalar-v0`: scalar head ([D33](/design/d33-scalar-per-option-head.md)), chat format ([D34](/design/d34-chat-format-for-slot-and-scalar.md)), sampled negatives (6 rows per choice decision), mixture v0 (371,275 decisions), `max_len` 2048, learning-rate scaling off, seed 0, one epoch, micro-batch 2 x 16. Compared with `baseline-v0` (letter baseline, frozen head, same mixture, context, seed and learning rates) and the other pilot models. Calibrated, one seed each, the same frozen evaluation sets. The Mimir overlap list is present for this run (validation was evaluated on `mixture-v0`).

| Set | Measure | `baseline-v0` | `slot-v0` | **`scalar-v0`** |
|---|---|---|---|---|
| Validation (6,127) | accuracy / NLL | 0.835 / 0.456 | 0.831 / 0.455 | **0.832 / 0.462** |
| | Brier / ECE | 0.201 / 0.016 | 0.202 / 0.018 | 0.204 / 0.024 |
| | noul / choice / score accuracy | 0.922 / 0.852 / 0.677 | 0.905 / 0.852 / 0.671 | 0.912 / 0.847 / 0.689 |
| | seen by Mimir (5,872) / not flagged (255) accuracy | 0.835 / 0.835 | 0.833 / 0.804 | 0.831 / 0.859 |
| Held-out nb-NO (600) | accuracy / NLL | 0.952 / 0.148 | 0.960 / 0.144 | 0.965 / 0.123 |
| **Extra, unseen label sets (32 sources, 7,323)** | accuracy / NLL | 0.812 / 0.476 | 0.804 / 0.480 | **0.807 / 0.513** |
| | gain over zero-shot (0.722) | +8.9 | +8.2 | +8.4 |
| Extra, 23 task-novel sources (5,565) | accuracy / gain | 0.791 / +7.2 | 0.781 / +6.2 | 0.796 / +7.6 |
| Typed Decisions test (descriptive) | accuracy / KL / Brier | 0.788 / 0.087 / 0.048 | 0.788 / 0.086 / 0.048 | 0.784 / **0.078 / 0.044** |
| Cost | training time, peak memory | 15.8 h, 33.6 GB | - | **51.1 h, 48.7 GB** |

On the 32 extra sources the scalar head is better than `baseline-v0` on 10, worse on 19, tied on 3 (against `baseline-lmhead` 18 / 13 / 1, against `slot-v0` 13 / 15 / 4). Largest drops against `baseline-v0`: `open_jev__email-selection-control` -14.6 points, `corr2cause` -9.5, `logical_entailment` -9.0, the security workflow control -5.5; largest gains: NanoNQ +5.8, NanoSCIDOCS +4.9, NanoHotpotQA +4.8, `citation-control` +3.0 (the three NanoBEIR sets are retrieval tasks with many candidates; not verified that the option count explains it).

## Interpretation

- **E17 rule (fixed beforehand):** the E12 gain over zero-shot is +8.4 for the scalar head against +8.9 for the letter baseline, a difference of -0.5 points (needed +2.0), and it wins on 10 of 32 sources (31%, needed 60%); validation accuracy is 0.3 points lower (limit 1.0). **Result: a tie; the simpler and cheaper letter baseline is preferred.** The swapped test (baseline supported) fails too (0.5 against 2.0 points).
- **With the pilot results this makes four readouts that tie on quality:** letter with frozen head, letter with trainable head, slot ([O26](/observations/o26-unseen-label-sets-slot-and-baseline-tie.md)) and scalar. Fine-tuning gives the gain, not the readout (E13).
- **Costs and calibration:** the scalar head took 3.2 times the training time (51 against 16 hours) and 45% more memory. Its calibration is slightly worse: validation ECE 0.024 against 0.016, and on the extra set the NLL is 0.037 higher (0.513 against 0.476) at about the same accuracy.
- **Descriptive only (not part of the rule, D11):** on Typed Decisions the scalar head has the lowest KL of all models so far (0.078 against 0.087 for `baseline-v0` and 0.081 for the 1M letter model) and the lowest Brier (0.044), with accuracy 0.784. It is a single run on one benchmark with one seed and decides nothing.
- **What the scalar head offers instead:** no limit of 26 options and a standard design; not tested here because the mixture caps options at 26.
- **Limits:** one seed per model; the scalar run also differs from the letter baseline in the sampled negatives, the per-option chat rows and the micro-batch composition; evaluation sets with n of 255 or 600 carry about 2 points of standard error.

## Reproduce

`python -m mimir_decide.compare runs/letter-zeroshot runs/baseline-v0 runs/baseline-lmhead runs/slot-v0 runs/scalar-v0 --file validation.json` (also `heldout.json`; `extra.json` with `--sources-file configs/extra_eval_sources.yaml` and optionally `--sources-key task_novel_sources`); figure `docs/figures/e17-scalar-v0-vs-baseline-v0.png`.

## Related

* [E17](/experiments/e17-scalar-head-against-letter-baseline.md)
* [D33](/design/d33-scalar-per-option-head.md)
* [O26](/observations/o26-unseen-label-sets-slot-and-baseline-tie.md)
* [Q04](/questions/q04-slot-vs-letter-on-real-model.md)
