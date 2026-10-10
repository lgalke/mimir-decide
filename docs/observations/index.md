# Observations

Dated, evidence-backed things seen so far; add new ones with the template.

## Observation

* [O01: 46% of the first 4,000 tasksource train rows are excluded by licence](/observations/o01-tasksource-licence-exclusion-rate.md) - Trial build: unknown, non-commercial or non-allowlisted licences remove nearly half of sampled tasksource rows.
* [O02: About a quarter of sampled bekko rows exceed 12,000 characters](/observations/o02-bekko-long-states.md) - Retrieval-style bekko tasks are mostly dropped by the length limit.
* [O03: Dataset cards and the data disagreed in several places](/observations/o03-dataset-cards-differ-from-the-data.md) - Schemas and split labels had to be verified by reading real rows.
* [O04: HelpSteer2's own train and validation overlap slightly](/observations/o04-helpsteer2-upstream-train-validation-overlap.md) - 5 of 1,038 validation responses also appear in upstream train; 2 prompts do.
* [O05: The same tasks arrive via several collections](/observations/o05-cross-collection-duplicates.md) - bekko and tasksource carry twins of cladder and corr2cause; aqua_rat overlaps math_qa.
* [O06: Within-source n-gram checks wrongly dropped 1,345 LocalLLaMA decisions](/observations/o06-localllama-template-overlap-false-positives.md) - Synthetic templated data repeats scenario text across its own train and test.
* [O07: Mimir v1.5 trained on tasksource and FLAN-style data](/observations/o07-mimir-policy-includes-tasksource-and-flan.md) - The sampling policy lists tasksource__, flan__, flan_factual__, posttrain_natural_instructions__ and sapient-synth-flan-*.
* [O08: Name-level audit flags almost everything](/observations/o08-audit-flag-rates.md) - 316/316 tasksource, 7/7 bekko, 5/5 HelpSteer2, 2/2 MASSIVE; only LocalLLaMA (0/4) is unflagged.
* [O09: Toy task: slot learns shuffled options quickly, letter baseline does not](/observations/o09-toy-slot-vs-letter.md) - 1-layer tiny HRM, 3 options, label from a keyword in the state.
* [O10: Trial build statistics (--limit 4000)](/observations/o10-trial-build-statistics.md) - Counts and timing of the clean trial build used for smoke tests.
* [O11: Segment-wise tokenization changed '\n\n' into two tokens](/observations/o11-letter-prompt-tokenization-boundary.md) - The letter prompt differed from the tokenizer's own encoding until boundaries were fixed.
* [O12: Smoke-run metrics on the random tiny model are meaningless](/observations/o12-uncalibrated-tiny-model-smoke-numbers.md) - Do not read the tiny-model accuracy, NLL or temperatures as results.
* [O13: L_bp_cycles [0,3] shrinks L-stack gradients about 230x](/observations/o13-l-bp-cycles-changes-gradient-scale.md) - Single-batch probe on the tiny model: the L stack gets far smaller gradients with [0,3] than with the checkpoint's [3,3].
* [O14: First pilot results: slot readout and letter baseline are tied](/observations/o14-first-pilot-results-slot-and-baseline-tie.md) - One seed, validation n=6127. Slot and letter baseline differ by at most 0.004 in accuracy-level metrics; 96% of validation comes from sources Mimir saw.
* [O15: Pilot training fits in 33.6 GB and runs at 6.15 examples per second](/observations/o15-pilot-training-memory-and-speed.md) - Memory and speed of the first real training run (slot model, default settings) on an NVIDIA RTX PRO 6000 Blackwell Server Edition.
* [O16: The untrained model reaches 0.669 accuracy; fine-tuning adds about 17 points](/observations/o16-zero-shot-baseline-fine-tuning-adds-16-points.md) - Zero-shot Mimir v1.5 with the letter prompt scores 0.669 on validation against 0.835 for the fine-tuned baseline. The gain is largest on Score and smallest on the MASSIVE held-out task.
* [O17: Calibration and selective risk of the pilot and zero-shot runs](/observations/o17-calibration-and-selective-risk-of-the-pilot-runs.md) - Fine-tuned models are well calibrated and slightly underconfident; their confidence ranks errors very well. The zero-shot model is overconfident, most on Score. Gaps in what is recorded are listed.
* [O18: Temperature scaling barely changes the fine-tuned models](/observations/o18-temperature-scaling-barely-changes-the-fine-tuned-models.md) - Raw and calibrated results of the slot model and the baseline differ by 0.001 to 0.009 in ECE and by at most 0.005 in NLL. Calibration does not help on the held-out task, and it worsens noul ECE.
* [O19: Neither model is sensitive to option order](/observations/o19-option-order-does-not-matter-for-either-model.md) - With 3 random option orders, validation accuracy changes by 0.03 points (slot) and 0.17 points (baseline). The pre-set rule does not support the slot design's order-invariance advantage.

## Template

* [Template: observation](/observations/template-observation.md) - Copy to observations/oNN-slug.md to record something seen, with evidence.
