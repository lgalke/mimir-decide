---
type: Observation
title: 'O24: The slot probe extracts little from a frozen backbone and is far worse than the letter probe'
description: A slot head trained on the frozen Mimir backbone reaches 0.446 validation accuracy, below the untrained letter readout (0.669) and the letter probe (0.703). The slot readout needs a trained backbone.
date: '2026-10-14'
confidence: medium
tags: [e13, probe, frozen-backbone, slot, results]
timestamp: 2026-10-14T00:00:00Z
---

## Observation

Source: `runs/slot-probe/` (E13: slot model, frozen backbone and embeddings, marker vector fixed at the mean embedding, head learning rate 1e-3 fixed in advance, pilot data and settings, one seed). Only the slot head trains (about 5,000 parameters).

| Calibrated, validation (n=6127) | Zero-shot | Letter probe | **Slot probe** | Slot (fine-tuned) |
|---|---|---|---|---|
| Accuracy | 0.669 | 0.703 | **0.446** | 0.831 |
| NLL | 0.806 | 0.766 | **1.302** | 0.455 |
| Brier / ECE | 0.376 / 0.032 | 0.354 / 0.013 | **0.610 / 0.032** | 0.202 / 0.018 |
| Noul accuracy (n=1079) | 0.776 | 0.792 | **0.619** | 0.905 |
| Choice accuracy (n=4030) | 0.714 | 0.713 | **0.396** | 0.852 |
| Score accuracy (n=1018) | 0.377 | 0.568 | **0.459** | 0.671 |
| Not flagged, LocalLLaMA (n=255) accuracy | 0.584 | 0.604 | **0.510** | 0.804 |
| Held-out nb-NO (n=600) accuracy / ECE | 0.895 / 0.047 | 0.905 / 0.030 | **0.487 / 0.221** | 0.960 / 0.027 |

**Typed Decisions benchmark** (calibrated, 2,000 decisions): accuracy 0.504, KL from gold 0.369, Brier 0.201 (untrained: 0.607, 0.295, 0.166; letter probe 0.611, 0.301, 0.164). Per kind accuracy: noul 0.622, choice 0.577, score 0.362.

**Training:** 11,603 steps, 3.0 h, 34.8 examples/s, peak memory 9.22 GB. The validation curve (2,000-decision sample) is flat from the first check: NLL 1.192 at step 1,000, 1.185 at step 8,000, 1.184 at the end (best 1.183); accuracy stays between 0.41 and 0.44. The head learns what it can within the first thousand steps.

## Interpretation

- **E13 rule (fixed beforehand):** the slot probe is **25.7 points worse** than the letter probe on validation accuracy (0.446 against 0.703), far beyond the 2-point margin. By the rule this is "the opposite": the slot readout does not extract more from the same frozen features; it extracts less. The slot probe is even below the untrained letter readout.
- **Consequence for D05:** the slot design works only together with a trained backbone. Full fine-tuning gives 0.831, so the backbone learns to put option information into the marker states. This is not evidence against the slot model as trained in the pilot; it removes the argument that the pretrained Mimir states are already suited to a slot readout.
- **Why it is weak (hypothesis, not tested):** all markers start from the same embedding, and a pretrained next-token model has no reason to make those positions informative. The letter readout uses the position and tokens that the model already uses for answers. The head has about 5,000 parameters on features that were not made for it.
- **Held-out nb-NO:** accuracy 0.487 with ECE 0.221 and mean confidence 0.266, so the post-hoc temperatures (0.82 to 1.67 on the benchmark) do not transfer; the slot probe is under-confident there.
- **Limits:** one seed; learning rate 1e-3 not tuned (the flat curve suggests that more steps would not help, but a different rate or a trainable marker vector was not tried); the marker vector was fixed, which handicaps this probe by design ([E13](/experiments/e13-head-only-baselines.md)). Read the result as "slot head on frozen features, as run", not as a statement about every frozen-feature slot readout.

## Reproduce

`python -m mimir_decide.compare runs/letter-zeroshot runs/letter-probe runs/slot-probe runs/baseline-v0 runs/slot-v0 --file validation.json`; benchmark numbers from `runs/slot-probe/eval/typed_decisions_test.json`; curve from `runs/slot-probe/train_log.jsonl`.

## Related

* [E13](/experiments/e13-head-only-baselines.md)
* [O23](/observations/o23-letter-probe-recovers-a-fifth-of-the-fine-tuning-gain.md)
* [D05](/design/d05-slot-readout-heads.md)
