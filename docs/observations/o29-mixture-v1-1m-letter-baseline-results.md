---
type: Observation
title: 'O29: The 1M letter baseline on mixture v1 improves validation and calibration but is worse on unseen label sets'
description: Against the pilot (baseline-v0) the 1M model gains 1.0 point of validation accuracy and 0.03 NLL, a little on Typed Decisions KL, and loses 1.7 points on the 32 unseen-label-set sources. By the E16 rule it is partly better.
date: '2026-10-24'
confidence: medium
tags: [e16, mixture-v1, results, unseen-labels, benchmark]
timestamp: 2026-10-24T00:00:00Z
---

## Observation

Run `runs/baseline-v1-1M` (letter baseline, frozen head, mixture v1 with 999,906 decisions, `max_len` 4096, 31,248 steps, 39.3 h at 7.06 decisions/s, peak 47.4 GB, learning-rate scaling off, seed 0, one epoch) against the pilot `baseline-v0` (mixture v0 with 371,275 decisions, `max_len` 2048) and the other pilot models. Calibrated, one seed each, `best` checkpoint (best validation NLL 0.4484 at step 30,000). The evaluation files are identical in both mixtures.

| Set | Measure | `baseline-v0` | **`baseline-v1-1M`** |
|---|---|---|---|
| Validation (6,127) | accuracy / NLL | 0.835 / 0.456 | **0.845 / 0.426** |
| | Brier / ECE | 0.201 / 0.016 | 0.187 / 0.015 |
| | noul / choice / score accuracy | 0.922 / 0.852 / 0.677 | 0.922 / 0.865 / 0.685 |
| Held-out nb-NO (600) | accuracy / NLL | 0.952 / 0.148 | 0.955 / 0.126 |
| **Extra, unseen label sets (32 sources, 7,323)** | accuracy / NLL | **0.812** / 0.476 | **0.795** / 0.524 |
| | gain over zero-shot (0.722) | +8.9 | +7.3 |
| Extra, 23 task-novel sources (5,565) | accuracy / NLL | 0.791 / 0.535 | 0.769 / 0.603 |
| Typed Decisions test (2,000) | accuracy / KL / Brier | 0.788 / 0.087 / 0.048 | 0.789 / 0.081 / 0.045 |
| | accuracy by kind (noul / choice / score) | 0.862 / 0.750 / 0.762 | 0.857 / 0.742 / 0.775 |
| | accuracy by workflow (agent trace / customer service / invoice / security) | 0.720 / 0.826 / 0.832 / 0.776 | 0.706 / 0.854 / 0.824 / 0.774 |

On the extra set the 1M model is better on 14 of 32 sources, worse on 16, tied on 2. Largest drops: `open_jev__entity-alignment-control` -18.5 points, `drone-control` -10.5, `email-selection-control` -9.6, `painting-geometry` -7.0, NanoArguAna -5.4; largest gains: the security-incidents workflow control +5.0, `followir_core17` +2.8, `contract_nli` +2.5. The other pilot models on the extra set: `baseline-lmhead` 0.802, `slot-v0` 0.804 (the 1M model is below all three).

## Interpretation

- **E16 rule (fixed beforehand), composition and size together:** validation accuracy rose by 0.98 points (needed 1.0, so just short, with NLL 0.030 lower) and the E12 gain over zero-shot **fell** by 1.6 points (needed +1.0). The rule's second condition fails clearly: **partly better** (in-distribution quality and calibration), **not better on unseen label sets.** Size, composition, the 4096 context and the number of steps (31,248 against 11,603) changed together, so none of them can be blamed.
- **The loss on unseen label sets is larger than the noise:** pooled n = 7,323 gives about 0.5 points of standard error per model, and the difference is 1.7 points; the per-source picture is mixed (14 up, 16 down) with a few large drops on synthetic `open_jev` controls. One seed.
- **Typed Decisions:** accuracy is the same (0.789), KL is 7% lower (0.081 against 0.087) and Brier 0.045 against 0.048; the score kind gains 1.3 points, customer service 2.8. Still below the published LoRA rows (0.79 to 0.80 accuracy, KL 0.080 to 0.081 are level) and far below the 150M specialist (0.906).
- **Hypotheses, not tested:** (1) three times more steps on the same kind of label sets specialise the model to the training label sets; (2) all runs so far have the learning-rate scaling off, i.e. an L stack at 6 times and an H stack at 2 times Mimir's last rate ([O28](/observations/o28-mimir-basics-check.md)), which would act more strongly with more steps; (3) the kind and option-count weights of the new mixture shift what is learned. Candidate checks: the same run with `lr_cycle_dividers` on, a v1-371k run at the same step count as v0.
- **Calibration:** ECE is 0.015 (0.0148) on validation, as good as before; on the extra set the confidence (0.808) matches the accuracy (0.795) well.
- **Evaluation caveat (my omission):** the 1M run's validation file has no `sources_seen_by_mimir` list, because the Mimir overlap audit was never run on `mixture-v1` (the pilot script ran it). `compare` therefore shows the whole validation set as 'not flagged' for this run. The numbers above are unaffected; the seen/unflagged split for the 1M run needs `audit_mimir_overlap` on `mixture-v1` and a re-evaluation of validation (see the log). Every source of the extra set is flagged as seen by Mimir (O26), the validation set is 96% flagged (O14).

## Reproduce

`python -m mimir_decide.compare runs/letter-zeroshot runs/baseline-v0 runs/baseline-v1-1M --file validation.json` (also `heldout.json`, and `extra.json` with `--sources-file configs/extra_eval_sources.yaml` and optionally `--sources-key task_novel_sources`); benchmark numbers from `runs/*/eval/typed_decisions_test.json`.

## Related

* [E16](/experiments/e16-mixture-v1-training.md)
* [O27](/observations/o27-mixture-v1-build-results.md)
* [O26](/observations/o26-unseen-label-sets-slot-and-baseline-tie.md)
* [Q16](/questions/q16-composition-or-size.md)
