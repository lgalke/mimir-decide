---
type: Observation
title: 'O32: The Clef-style probe is the best frozen head so far but still loses on unseen label sets; the fully fine-tuned Clef head ties'
description: On a frozen backbone the Clef-style head reaches 0.717 validation accuracy (letter probe 0.701), 0.703 and KL 0.163 on Typed Decisions, and 0.690 on the E12 sources (span probe 0.631, letter probe 0.732). Fully fine-tuned it ties with the other readouts.
date: '2026-11-02'
confidence: medium
tags: [e19, clef, probe, results]
timestamp: 2026-11-02T00:00:00Z
---

## Observation

Runs `runs/clef-probe-v1` (frozen backbone, only the Clef-style head of [D38](/design/d38-clef-style-scoring-head.md) trains, head learning rate 3e-4) and `runs/clef-v1-371k-lrdiv` (the same head, whole model fine-tuned, learning-rate dividers on, EMA 0.999 which made no difference), both on mixture `v1-371k`, `max_len` 4096, chat format, effective batch 32, 11,599 steps, seed 0, calibrated. The probes ran 3.3 h (Clef), 3.0 h (letter) and 2.9 h (span) at 11.6 to 11.8 GB; the full run 15.7 h at 6.57 decisions/s, 51.4 GB.

**Probes (frozen backbone)**

| Set | Measure | zero-shot | `letter-probe-v1` | `span-probe-v1` | **`clef-probe-v1`** |
|---|---|---|---|---|---|
| Validation (6,127) | accuracy / NLL | 0.669 / 0.806 | 0.701 / 0.765 | 0.700 / 0.788 | **0.717 / 0.719** |
| | Brier / ECE | 0.376 / 0.032 | 0.353 / 0.018 | 0.366 / 0.022 | 0.337 / 0.017 |
| | noul / choice / score accuracy | 0.776 / 0.714 / 0.377 | 0.794 / 0.710 / 0.567 | 0.796 / 0.706 / 0.577 | 0.803 / 0.726 / 0.589 |
| Held-out nb-NO (600) | accuracy / NLL | 0.895 / 0.310 | 0.895 / 0.309 | 0.908 / 0.287 | **0.918 / 0.265** |
| **Extra, unseen label sets (7,323)** | accuracy / NLL | 0.722 / 0.638 | **0.732** / 0.631 | 0.631 / 0.870 | **0.690** / 0.748 |
| | gain over zero-shot | - | +1.0 | -9.1 | **-3.3** |
| Extra, 23 task-novel sources | accuracy | 0.719 | 0.732 | 0.619 | 0.688 |
| Typed Decisions test (descriptive) | accuracy / KL / Brier | 0.607 / 0.295 / 0.166 | 0.593 / 0.295 / 0.163 | 0.696 / 0.208 / 0.113 | **0.703 / 0.163 / 0.089** |
| | noul / choice / score accuracy | 0.633 / 0.563 / 0.619 | 0.602 / 0.563 / 0.609 | 0.815 / 0.655 / 0.637 | 0.825 / 0.673 / 0.634 |

**Full fine-tuning (whole model)**

| Set | Measure | `baseline-v0` | `scalar-v0` | `baseline-v1-1M` | **`clef-v1-371k-lrdiv`** |
|---|---|---|---|---|---|
| Validation | accuracy / NLL | 0.835 / 0.456 | 0.832 / 0.462 | 0.845 / 0.426 | **0.827 / 0.460** |
| Held-out nb-NO | accuracy / NLL | 0.952 / 0.148 | 0.965 / 0.123 | 0.955 / 0.126 | 0.955 / 0.139 |
| Extra (32 sources) | accuracy / gain | 0.812 / +8.9 | 0.807 / +8.4 | 0.795 / +7.3 | **0.805 / +8.3** |
| Extra, task-novel (23) | accuracy / gain | 0.791 / +7.2 | 0.796 / +7.6 | 0.769 / +4.9 | 0.791 / +7.2 |
| Typed Decisions (descriptive) | accuracy / KL / Brier | 0.788 / 0.087 / 0.048 | 0.784 / 0.078 / 0.044 | 0.789 / 0.081 / 0.045 | 0.786 / 0.088 / 0.047 |

The full Clef run is better than `baseline-v0` on 14 of 32 extra sources and worse on 15 (3 ties); against `baseline-v1-1M` 15 better, 14 worse.

## Interpretation

- **E19 probe rule (fixed beforehand):** (1) against the letter probe on the same data the validation accuracy is +1.6 points (0.717 against 0.701), inside the 2-point margin: **no difference by the E13 margin**, although NLL (0.719 against 0.765), Brier and the choice kind are clearly better, so it is the best frozen head on our validation set. (2) The lexical-prior hypothesis was supported only if the E12 accuracy came within 1 point of the letter probe (0.732) with validation near the span probe: the Clef probe reaches **0.690, 4.2 points below the letter probe: not supported by the rule.** It recovers about 58% of the span probe's loss on unseen label sets (from -9.1 to -3.3 against +1.0 for the letter probe) but stays below the untrained model.
- **Where it still fails on E12:** large drops against the letter probe on `open_jev__sponsor-segment-control` (0.515 against 0.970), `ruletaker` (0.656 against 0.928), `reasoning-control` (0.640 against 0.865) and `ethics` (0.626 against 0.732); gains on `vizdoom-basic` (0.946 against 0.904), `citation-control` (0.780 against 0.720), `followir_core17` (+5). Several of the failures are tasks with few fixed labels, which fits a learned label prior, but this was not tested.
- **On Typed Decisions the Clef probe is the strongest frozen head** (0.703 accuracy, KL 0.163, Brier 0.089; span probe 0.696 / 0.208 / 0.113; letter probe 0.593 / 0.295), still far from the fine-tuned models (0.78 to 0.79, KL 0.078 to 0.088). Descriptive only (D11). This is the nearest to a replication of the PhD student's heads-only result, if she trained only heads.
- **Fully fine-tuned the Clef head ties** with the other readouts on every set (validation 0.827 against 0.835, extra gain +8.3 against +8.9, Typed Decisions 0.786 with KL 0.088). **The E19 full-model rule cannot be applied as written:** it names the matched runs `span-v1-371k-lrdiv` and `baseline-v1-371k-lrdiv`, which were not run (the 1M dividers runs were interrupted). The comparison with `baseline-v0` is confounded by mixture, context and the dividers; the pre-set rule is not changed after seeing results, so the full-model question stays open until the matched letter run exists (about 14 h).
- **EMA:** the averaged weights ended at NLL 0.4598 / accuracy 0.824 against 0.4566 / 0.823 on the 2,000-decision sample: no gain.
- **Limits:** one seed per model; head size and learning rate not tuned; the probe results at n = 7,323 carry about 0.5 points of standard error, per-source results (n about 200) about 3 points.

## Reproduce

`python -m mimir_decide.compare runs/letter-zeroshot runs/letter-probe-v1 runs/span-probe-v1 runs/clef-probe-v1 --file validation.json` (also `heldout.json`; `extra.json` with `--sources-file configs/extra_eval_sources.yaml`, optionally `--sources-key task_novel_sources`); benchmark numbers from `runs/*/eval/typed_decisions_test.json`.

## Related

* [E19](/experiments/e19-clef-style-head.md)
* [D38](/design/d38-clef-style-scoring-head.md)
* [O31](/observations/o31-span-probe-matches-the-letter-probe-in-distribution-but-fails-on-unseen-label-sets.md)
