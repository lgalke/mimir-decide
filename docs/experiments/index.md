# Experiments

Prefilled experiment plans: hypothesis, exact commands, metrics, decision rule, results.

## Experiment

* [E01: Full mixture build](/experiments/e01-full-mixture-build.md) - [running] Build the real mixture and record what survives licences, leakage guards and caps.
* [E02: Smoke run on the real model](/experiments/e02-smoke-run-real-model.md) - [done] Load Mimir v1.5, train 50 steps, measure memory and throughput, confirm loss decreases and checkpoints reload.
* [E03: Slot readout vs letter baseline (pilot)](/experiments/e03-slot-vs-letter-pilot.md) - [done] The central comparison: same data, backbone, loss and calibration; only the readout differs.
* [E04: Effect of temperature calibration](/experiments/e04-calibration-effect.md) - [done] Quantify ECE before and after per-kind temperature scaling, on validation and held-out tasks.
* [E05: Option-order robustness](/experiments/e05-option-order-robustness.md) - [done] Evaluate with options presented in random orders and measure the change in accuracy and agreement.
* [E06: L_bp_cycles ablation](/experiments/e06-l-bp-cycles-ablation.md) - [planned] Compare `[3,3]` (checkpoint) with `[0,3]` for memory, speed and quality.
* [E07: Seen vs not-flagged analysis](/experiments/e07-contamination-split-analysis.md) - [planned] Separate every result by whether Mimir's mixture contains the source (name-level).
* [E08: Latency and throughput](/experiments/e08-latency-and-throughput.md) - [planned] Measure decisions per second and ms per decision for the slot model against Jev's claims.
* [E09: Recurrence depth probe (exploratory)](/experiments/e09-recurrence-depth-probe.md) - [planned] Evaluate the trained model with fewer H cycles to see whether depth matters for decisions.
* [E10: Zero-shot letter baseline (untrained Mimir)](/experiments/e10-zero-shot-letter-baseline.md) - [done] Evaluate the untrained Mimir v1.5 with the letter prompt, to show how much of the fine-tuned baseline's quality was already there.
* [E11: Stronger baseline with a trainable language-model head](/experiments/e11-stronger-baseline-trainable-lm-head.md) - [done] Repeat the letter baseline with the LM head unfrozen, so the baseline is not handicapped before it is compared with the slot model.
* [E12: Unseen label sets (evaluation on additional tasks)](/experiments/e12-unseen-label-sets.md) - [done] Evaluate all existing models, without retraining, on extra tasks whose label sets never occur in the training mixture.
* [E13: Head-only baselines (frozen backbone)](/experiments/e13-head-only-baselines.md) - [done] Train only the output head on the frozen Mimir backbone, for the letter readout (head initialised from the LM head) and for the slot readout. Shows how much of the fine-tuning gain a readout alone can reach.
* [E14: Typed Decisions benchmark (comparison with Anna's approach)](/experiments/e14-typed-decisions-benchmark.md) - [done] Score our models on the full test split of LocalLLaMA/typed-decisions (400 cases, 2,000 decisions) with accuracy, KL from gold, Brier and ECE, to compare with a second approach on the same data.
* [E15: Decision Index public suite](/experiments/e15-decision-index-public-suite.md) - [planned] Run our models through the public suite of the Decision Index (37 benchmarks, about 140k requests) with the kit's own scorer, after checking for overlap with our training mixture.
* [E16: Training on mixture v1 (composition, size and 4096 context)](/experiments/e16-mixture-v1-training.md) - [running] Train the slot model and the letter baseline on the larger, more diverse mixture v1, first at v0's size, then larger, and compare on the frozen evaluation.
* [E17: Scalar per-option head against the letter baseline](/experiments/e17-scalar-head-against-letter-baseline.md) - [done] Train the scalar head (D33) on mixture v1 and compare it with the letter baseline on the frozen evaluation, with the cost per example next to the quality.
* [E18: Span-pooling readout with per-kind heads, frozen and fully fine-tuned](/experiments/e18-span-pooling-probe-and-fine-tuning.md) - [running] A head-only probe and a full fine-tuning run of the span-pooling readout (D35) on the v0 mixture, compared with the E13 probes and with baseline-v0.
* [E19: Clef-style head, frozen and fully fine-tuned](/experiments/e19-clef-style-head.md) - [planned] Test the Clef-style scoring head (D38): a head-only probe on mixture v1, compared with the span and letter probes, and a full fine-tuning run.

## Template

* [Template: experiment](/experiments/template-experiment.md) - Copy to experiments/eNN-slug.md to plan or record an experiment.
