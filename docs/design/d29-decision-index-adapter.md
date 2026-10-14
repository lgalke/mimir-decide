---
type: Decision
title: 'D29: Decision Index adapter: our models as a kit engine, refusing instead of truncating'
description: mimir_decide.decision_index_engine maps kit requests to our decisions and returns calibrated probabilities. A request that does not fit is refused as unsupported; nothing is cut.
status: accepted
date: '2026-10-14'
decided_by: owner (add the benchmark); mapping by agent
tags: [evaluation, benchmark, decision-index]
timestamp: 2026-10-14T00:00:00Z
---

## Context

The owner asked to add the Decision Index ([research note](/research/decision-index.md)). The kit evaluates any `Engine` and forbids truncation, option pruning and per-benchmark prompt changes. Our formatter truncates long states from the middle and cuts options at 160 tokens, which is allowed in training and in our own evaluation but not in the Index.

## Decision

- **Adapter:** `mimir_decide/decision_index_engine.py`, class `MimirDecideEngine`, used as `--engine mimir_decide.decision_index_engine:MimirDecideEngine --option run_dir=runs/<run> --option checkpoint=best|zero-shot`.
- **Mapping:** a `choice` question is a Choice decision; its options are the criteria descriptions in the given order, its question is the instructions (JSON objects are serialised compactly), the answer is the argmax of the softmax. A `noul` question is a Noul decision with options "yes" and "no" and the answer is P(yes), as in training. The state is used unchanged. The slot and letter models both work; the prompt is the training prompt, the same for every benchmark.
- **Calibration:** the per-kind temperatures of the run's `calibration.json` are applied by default (`calibrated=false` switches it off). Temperature does not change the argmax, so accuracy-type metrics are unaffected.
- **Refuse, never cut:** the adapter raises `Unsupported` when the full prompt exceeds `max_len` (default: the value in the checkpoint, 2,048), when an option has more than 160 tokens, or when the letter model gets more than 26 options. The kit counts these as wrong. The slot model has no option limit besides the length.
- **No test-split issue:** the Index rows are not our test split; they are an external benchmark. The leakage check against our training mixture is part of E15.

## Consequences

- The Index measures what our models do outside their training distribution, with coverage penalties. The letter baseline will have lower coverage than the slot model on benchmarks with more than 26 options (for example ToolRet and BRIGHT with 32 candidates); report coverage next to every score.
- Raising `max_len` to 4,096 (the model's limit, above the 2,048 used in training) is a second, separately reported setting, never silently chosen per benchmark.
- Tested offline with fake models and against the kit's `validate` (the kit installed in a scratch folder: its `Engine` and `Unsupported` classes are used when present). Not run on the real suite.
- Publishing a submission needs the owner's permission and public weights (we have published nothing).

## Revisit when

The kit changes its engine interface, or the owner wants a native batch path to meet the 1,000 ms latency rule.

## Related

* [E15](/experiments/e15-decision-index-public-suite.md)
* [D28](/design/d28-typed-decisions-scorer-and-test-use.md)
* [D11](/design/d11-split-isolation-and-test-access.md)
