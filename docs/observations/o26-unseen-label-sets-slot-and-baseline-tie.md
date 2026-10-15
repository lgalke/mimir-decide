---
type: Observation
title: 'O26: On tasks with unseen label sets, slot and letter baseline tie (E12)'
description: On 32 evaluation-only bekko sources (7,323 decisions) the fine-tuned models gain 8 to 9 points over the untrained model. By the rule fixed beforehand the slot model and the baseline tie. All sources are flagged as seen by Mimir.
date: '2026-10-16'
confidence: medium
tags: [e12, unseen-labels, slot, baseline, results]
timestamp: 2026-10-16T00:00:00Z
---

## Observation

Source: `runs/*/eval/extra.json` for the six runs below, produced by `evaluate --split extra` on the evaluation-only set built by `build_extra_eval` (D26 option A): 32 bekko sources, 7,323 decisions, source list committed before the evaluation (`configs/extra_eval_sources.yaml`, commit `50dd7f5`). Calibrated with each run's existing temperatures. One seed per model. **All 32 sources are flagged as seen by Mimir by name**; they are unseen only for our fine-tuning. Reproduce: `python -m mimir_decide.compare runs/letter-zeroshot runs/letter-probe runs/slot-probe runs/baseline-v0 runs/baseline-lmhead runs/slot-v0 --file extra.json --sources-file configs/extra_eval_sources.yaml` (add `--sources-key task_novel_sources` for the 23-source pool).

| Pooled, 32 sources (n=7,323) | Accuracy | Gain vs zero-shot (points) | NLL | Brier | ECE |
|---|---|---|---|---|---|
| Untrained Mimir (letter prompt) | 0.722 | - | 0.638 | 0.354 | 0.027 |
| Letter probe | 0.735 | +1.3 | 0.626 | 0.348 | 0.025 |
| Slot probe | 0.374 | -34.8 | 1.205 | 0.672 | 0.176 |
| Letter baseline, frozen head (`baseline-v0`) | 0.812 | +8.9 | 0.476 | 0.252 | 0.024 |
| Letter baseline, trainable head | 0.802 | +8.0 | 0.490 | 0.263 | 0.026 |
| Slot model (`slot-v0`) | 0.804 | +8.2 | 0.480 | 0.254 | 0.029 |

Pooled over the 23 task-novel sources (n=5,565): zero-shot 0.719; slot 0.781 (+6.2); frozen-head baseline 0.791 (+7.2); trainable-head baseline 0.785 (+6.5).

By kind (accuracy / ECE): noul (n=4,444) slot 0.836 / 0.040, baseline 0.852 / 0.021; choice (n=1,554) slot 0.848 / 0.071, baseline 0.849 / 0.040; score (n=1,325) slot **0.644 / 0.050**, baseline 0.632 / 0.107, zero-shot 0.537 / 0.123.

Per source (n about 200, so one standard error is about 3 points): the slot model beats the frozen-head baseline on 11 of 32 sources, loses on 16, ties on 5; against the trainable-head baseline 17 wins, 9 losses; against both 10 of 32. Largest slot gains over the frozen-head baseline: `open_jev__citation-control` (+12.5 points), `painting-geometry` (+8.0), `security_incidents` workflow (+5.0). Largest losses: `drone-control` (-22.5), `vizdoom-basic` (-18.1), `context-retention` (-18.0), `logical_entailment` (-5.0).

## Interpretation

- **E12 rule (fixed beforehand):** `G_slot - G_best_baseline` = 8.2 - 8.9 = **-0.7** points on all 32 sources, below the +2.0 needed for 'slot supported'; the swapped test (+0.7) is also below 2.0. The slot model wins on only 34% of sources against the frozen-head baseline (rule: 60%). On the task-novel pool the difference is -1.0 (reversed). **Result: a tie; the simpler baseline is preferred.**
- **The design bet is not supported on unseen label sets.** The open-vocabulary argument for the slot readout does not show up here: fine-tuning gives both readouts the same gain, and the largest per-source differences go in both directions, which looks like source-specific variance with one seed.
- **Score questions are the one hint for the slot model:** accuracy 0.644 against 0.632 and clearly better calibration (ECE 0.050 against 0.107, NLL 0.853 against 0.929). This was not a pre-registered comparison, n = 1,325 and one seed; treat it as a lead for a later experiment, not a result.
- **Fine-tuning helps outside the training label sets:** +8 to +9 points over the untrained model, smaller on the task-novel pool (+6 to +7). The probes do not help (letter probe +1.3, slot probe far below chance level on score questions: 0.137).
- **Limits:** one seed per model; all sources flagged as seen by Mimir, 18 of 32 are `open_jev__*` synthetic control tasks; four of them have the names of the Typed Decisions workflows and were not compared with that benchmark's test rows; label novelty is exact-string, so 'unseen' means unseen wording (the tool's `familiar_task_in_training` flag marks `snli`, `corr2cause`, `ethics`, `logical_entailment` and `ruletaker` as tasks that also exist in training). `bekko/nlsat` is at chance for every model (0.49).

## Related

* [E12](/experiments/e12-unseen-label-sets.md)
* [D26](/design/d26-evaluate-unseen-label-sets-on-additional-tasks.md)
* [Q04](/questions/q04-slot-vs-letter-on-real-model.md)
* [O20](/observations/o20-trainable-output-head-ties-with-the-frozen-head-baseline.md)
