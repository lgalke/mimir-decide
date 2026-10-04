# Experiments

Prefilled experiment plans: hypothesis, exact commands, metrics, decision rule, results.

## Experiment

* [E01: Full mixture build](/experiments/e01-full-mixture-build.md) - [planned] Build the real mixture and record what survives licences, leakage guards and caps.
* [E02: Smoke run on the real model](/experiments/e02-smoke-run-real-model.md) - [planned] Load Mimir v1.5, train 50 steps, measure memory and throughput, confirm loss decreases and checkpoints reload.
* [E03: Slot readout vs letter baseline (pilot)](/experiments/e03-slot-vs-letter-pilot.md) - [planned] The central comparison: same data, backbone, loss and calibration; only the readout differs.
* [E04: Effect of temperature calibration](/experiments/e04-calibration-effect.md) - [planned] Quantify ECE before and after per-kind temperature scaling, on validation and held-out tasks.
* [E05: Option-order robustness](/experiments/e05-option-order-robustness.md) - [planned] Evaluate with options presented in random orders and measure the change in accuracy and agreement.
* [E06: L_bp_cycles ablation](/experiments/e06-l-bp-cycles-ablation.md) - [planned] Compare `[3,3]` (checkpoint) with `[0,3]` for memory, speed and quality.
* [E07: Seen vs not-flagged analysis](/experiments/e07-contamination-split-analysis.md) - [planned] Separate every result by whether Mimir's mixture contains the source (name-level).
* [E08: Latency and throughput](/experiments/e08-latency-and-throughput.md) - [planned] Measure decisions per second and ms per decision for the slot model against Jev's claims.
* [E09: Recurrence depth probe (exploratory)](/experiments/e09-recurrence-depth-probe.md) - [planned] Evaluate the trained model with fewer H cycles to see whether depth matters for decisions.

## Template

* [Template: experiment](/experiments/template-experiment.md) - Copy to experiments/eNN-slug.md to plan or record an experiment.
