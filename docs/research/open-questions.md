---
type: Reference
title: Open questions
description: Unknowns about Jev and about adapting Mimir; things to verify.
tags: [open-questions]
timestamp: 2026-10-03T00:00:00Z
---

# Open questions

> Project-level open questions now live one per page in [Open questions](/questions/index.md). This page keeps the research-level unknowns about Jev and HRM-Text from 2026-10-03.

- What is Jev's backbone and size? What is the "parallel sampler"? Public sources give no details ([Jev](/research/jev.md)).
- Is RLCD really RL, or proper-scoring-rule supervised training with RL branding?
- Does Mimir's recurrence hurt or help a pooled readout compared to a Transformer? ([HRM-Text](/research/hrm-text.md))
- Is a pre-instruction-tuning base checkpoint of Mimir available, and would it be a better start?
- Best way to get open-vocabulary Choice options with a fixed head.
- How well does calibration transfer across domains and to Danish?
- Can ACT be added to an already trained HRM-Text without retraining?
- Need to read the full Mimir paper (only abstract and HF card were read) and the Jev docs directly (only third-party summaries were read).

## Added 2026-10-04

- **Clean generalisation evidence.** Almost every available source is flagged as seen by Mimir (name-level, [dataset-selection](/design/dataset-selection.md)). What data can Mimir not have seen: newly written Danish items, or the deferred in-house sets (check their names against the v1.5 policy first: `giannor_dala_*`, `giannor_gec_*` and Arena data are in it)?
- **Row-level overlap** with Mimir's training data cannot be computed from public artifacts (DFM10 base is private). Could the Mimir team run a hash check against our eval files?
- **Real-model run.** Memory, speed and learning on the real 1.8B model are unmeasured; the scripts were exercised only on a tiny random HRM. Does `L_bp_cycles` `[3,3]` fit a 40 GB GPU with gradient checkpointing at 2048 tokens?
- **Does the slot readout beat the letter baseline on the real model?** Only a toy mechanism check exists ([heads-and-baseline](/design/heads-and-baseline.md)).
- **Licence.** Is a model trained on CC-BY-SA data itself bound by share-alike? Which bekko "qualified" subsets could be admitted after review?
- **bekko coverage.** Strict licence policy removes most subsets; how many decisions remain in a full build, and are they diverse enough?
- **Label semantics.** Soft labels (human disagreement, teacher probabilities) are not probabilities of correctness; calibration to them is not calibration to truth.
- **Dropped long states.** About a quarter of sampled bekko rows exceed 12,000 characters and are dropped; retrieval-style tasks are underrepresented as a result.
- **Teacher-label distillation** (stage 1 of the first plan) is not implemented.
