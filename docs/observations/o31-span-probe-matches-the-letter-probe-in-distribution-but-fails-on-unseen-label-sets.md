---
type: Observation
title: 'O31: The span-pooling probe matches the letter probe on validation, scores far higher on Typed Decisions and far lower on unseen label sets'
description: A frozen backbone with mean-pooled option text and per-kind heads reaches 0.700 validation accuracy (letter probe 0.701), 0.696 on Typed Decisions (letter probe 0.593) but only 0.631 on the E12 sources (zero-shot 0.722). By the E18 rule it ties on validation.
date: '2026-10-31'
confidence: medium
tags: [e18, probe, span, frozen-backbone, results]
timestamp: 2026-10-31T00:00:00Z
---

## Observation

Runs `runs/span-probe-v1` and `runs/letter-probe-v1` ([E18](/experiments/e18-span-pooling-probe-and-fine-tuning.md), [D35](/design/d35-span-pooling-readout-with-per-kind-heads.md)): frozen backbone, only the head trains (span: three per-kind linear heads on the mean of the final hidden states over each option's text; letter: a trainable copy of the 26 letter rows), mixture `v1-371k`, `max_len` 4096, chat format, learning rate 1e-3 for the head, 11,603 steps (2.9 h and 3.0 h), seed 0, calibrated. Earlier probes (E13) for reference: letter probe on v0 0.703, slot probe (marker readout) 0.446.

| Set | Measure | zero-shot | `letter-probe-v1` | **`span-probe-v1`** | slot probe (v0) |
|---|---|---|---|---|---|
| Validation (6,127) | accuracy / NLL | 0.669 / 0.806 | 0.701 / 0.765 | **0.700 / 0.788** | 0.446 / 1.302 |
| | ECE | 0.032 | 0.018 | 0.022 | 0.032 |
| | noul / choice / score accuracy | 0.776 / 0.714 / 0.377 | 0.794 / 0.710 / 0.567 | 0.796 / 0.706 / 0.577 | |
| Held-out nb-NO (600) | accuracy | 0.895 | 0.895 | 0.908 | 0.487 |
| **Extra, unseen label sets (7,323)** | accuracy / NLL | 0.722 / 0.638 | 0.732 / 0.631 | **0.631 / 0.870** | 0.374 / 1.205 |
| | gain over zero-shot | - | +1.0 | **-9.1** | -34.8 |
| Typed Decisions test (descriptive) | accuracy / KL / Brier | 0.607 / 0.295 / 0.166 | 0.593 / 0.295 / 0.163 | **0.696 / 0.208 / 0.113** | 0.504 / 0.369 / 0.201 |
| | noul / choice / score accuracy | 0.633 / 0.563 / 0.619 | 0.602 / 0.563 / 0.609 | 0.815 / 0.655 / 0.637 | |

For comparison the fully fine-tuned models reach 0.78 to 0.79 accuracy and KL 0.078 to 0.087 on Typed Decisions, and 0.83 to 0.845 on validation.

## Interpretation

- **E18 rule (probe, fixed beforehand):** against the letter probe on the same data the span probe is 0.1 points lower on validation accuracy (0.700 against 0.701), within the 2-point margin: **no difference.** The share of the zero-shot-to-fine-tuned gap that it recovers is R = (0.700 - 0.669) / (0.835 - 0.669) = 0.19, as for the letter probe (E13: below 0.25: the gain comes mostly from the backbone).
- **The marker was the problem of the slot probe:** replacing the untrained marker token by the mean over the option's own tokens lifts a frozen slot-style readout from 0.446 to 0.700 validation accuracy, to the level of the letter readout (O24).
- **On unseen label sets the span probe fails:** 0.631 on the 32 E12 sources, 9 points below the untrained model and 10 below the letter probe (E12 is not part of the E18 probe rule, which uses validation; it is the clearest negative result here). A hypothesis, not tested: a head over mean-pooled option text can learn which label strings tend to be correct (a prior over the training label vocabularies), which helps in distribution and on templated benchmarks and hurts for new label sets; the letter readout cannot store label identity.
- **On Typed Decisions the span probe is strong for a frozen backbone** (0.696 accuracy, KL 0.208 against 0.295, noul 0.815 against 0.602), although validation shows no gain. The benchmark has a few fixed workflows and label sets, and 3,404 of its train decisions are in the training mixture, so a head can learn them; the benchmark's 'prior' row (ignores the input) already reaches 0.470 accuracy ([O21](/observations/o21-benchmark-scorer-matches-the-published-uniform-row-only-for-kl-and-brier.md)). If the PhD student trained only heads, this is the kind of result to expect; whether she did is open. Descriptive only (D11).
- **Per-kind heads against a shared head** were not tested separately (`per_kind_heads=false` is available).
- **Limits:** one seed; the learning rate 1e-3 was fixed in advance and not tuned; the two probes share data, context and settings.

## Reproduce

`python -m mimir_decide.compare runs/letter-zeroshot runs/letter-probe-v1 runs/span-probe-v1 --file validation.json` (also `heldout.json`; `extra.json` with `--sources-file configs/extra_eval_sources.yaml`); benchmark numbers from `runs/*/eval/typed_decisions_test.json`.

## Related

* [E18](/experiments/e18-span-pooling-probe-and-fine-tuning.md)
* [D35](/design/d35-span-pooling-readout-with-per-kind-heads.md)
* [O24](/observations/o24-slot-probe-extracts-little-from-a-frozen-backbone.md)
* [O23](/observations/o23-letter-probe-recovers-a-fifth-of-the-fine-tuning-gain.md)
