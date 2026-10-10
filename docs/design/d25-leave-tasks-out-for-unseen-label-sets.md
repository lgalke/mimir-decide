---
type: Decision
title: 'D25: Leave whole tasks out to test unseen label sets'
description: E12 trains on a mixture derived from the pilot mixture without ten selected sources, chosen by a measurable label-novelty rule and pre-registered. The pilot configuration stays unchanged.
status: superseded
date: '2026-10-11'
decided_by: owner chose the experiment; design by agent, pending the owner's confirmation of thresholds and rule
tags: [evaluation, generalisation, held-out]
timestamp: 2026-10-11T00:00:00Z
---

> Superseded on 2026-10-11 by [D26](/design/d26-evaluate-unseen-label-sets-on-additional-tasks.md): retraining on a reduced mixture makes the runs incomparable with the pilot and with the stronger baseline. The tools (`select_heldout`, `derive_mixture`) stay as a fallback.

## Context

The owner called label sets unseen in training crucial for the slot design and asked for it as the next experiment. The pilot cannot test it: validation sources also occur in training, and the held-out task `massive/nb-NO` shares its intent names with the other MASSIVE locales ([D15](/design/d15-held-out-task-massive-nb-no.md)).

## Decision

- Hold out whole **sources** (tasks), not rows. Eligibility is measured: a fixed label set, almost no row whose option set another training source uses, few shared option strings ([E12](/experiments/e12-unseen-label-sets.md)). Selection is seeded, stratified by kind, and includes two sources that Mimir's mix does not contain by name.
- The selection file `configs/heldout_unseen_labels.yaml` is committed before any training (pre-registration).
- The new mixture is **derived** from the pilot mixture (`mimir_decide.derive_mixture`): train rows of the selected sources are removed, their eval rows move to the held-out file. No rebuild, no new downloads, and no new leakage, because only train rows are removed.
- `configs/data.yaml` and the pilot's held-out set stay as they are. The pilot results remain comparable, and the pilot models serve as the 'label sets seen' reference.
- The decision rule (2 points of pooled accuracy gain over zero-shot and 60% of sources) is fixed in E12 before training.

## Consequences

- Two more trainings (about 33 GPU hours) on a mixture that is a bit smaller than the pilot's.
- The measured novelty is exact-string; a semantically similar label set can still count as new.
- AGENTS.md rule 4 (do not change `held_out_sources`) still holds for `configs/data.yaml`; the extra held-out sources live in the derived mixture only.

## Revisit when

Too few sources qualify (then loosen the thresholds and log it), or the owner prefers other criteria or a human-written test set for truly new label sets.

## Related

* [D05](/design/d05-slot-readout-heads.md)
* [D11](/design/d11-split-isolation-and-test-access.md)
* [D15](/design/d15-held-out-task-massive-nb-no.md)
