---
type: Reference
title: Decision Index (community leaderboard for Jev reproductions)
description: What the Decision Index is, how it scores, what is public and private, and what it needs from an entrant. Read on 2026-10-14 from the Space and the kit.
tags: [benchmark, leaderboard, jev]
timestamp: 2026-10-14T00:00:00Z
---

# Decision Index

Sources: [Space](https://huggingface.co/spaces/multimodalart/jev-decision-index) (static; the README and `data/methodology.json` were read through the Hub API) and the MIT-licensed kit [apolinario/decision-index](https://github.com/apolinario/decision-index). Unofficial and community-maintained; not affiliated with TypeSafe. The page content is second-hand; check the Methodology page for rules before a submission.

## What it is

A leaderboard for open reproductions of Jev (typed decision engines). A request has a `state` and several typed `questions` (`choice` with criteria, or `noul` yes/no). The engine returns a probability for every option. Edition 0.3 (2026-10-06) is the current one.

- **Full score** = 20% public benchmarks + 50% private tests of the same skills + 30% private tasks from new domains, all on one scale. The private parts are run by the maintainers on submitted models and never published.
- **Public suite:** 37 benchmarks in five areas (Knowledge & Reasoning 25.8%, Language Understanding 25.8%, Retrieval & Classification 20.0%, Tools & Automation 18.3%, Arts & Human Taste 10%). About 110,201 requests (109,759 scoreable) plus 30,419 for seven later benchmarks. Examples: MMLU, GPQA Diamond, ARC, WinoGrande, HellaSwag, GSM8K, ANLI, BANKING77, CLINC150+OOS, BFCL, ToolRet, BRIGHT, Amazon ESCI, HLE, CLadder.
- **Scores** are chance-corrected (0 = random guessing, 100 = perfect) and coverage-adjusted (unanswered = wrong). Gold-star benchmarks weigh 1.2 inside their area. The metric differs per benchmark (accuracy, macro-F1, nDCG@10, per-review F1); different metrics are not averaged raw.
- **Rules for entrants:** no truncation (a request that does not fit counts as unsupported = wrong), no option pruning, no prompt tuning per benchmark, no labels in the adapter, no correctness-based retry. Native abstentions stand.
- **Latency:** the maintainers measure on one RTX PRO 6000; a median, mean or p80 above 1,000 ms per request keeps a model off the board.
- **Submission:** run the whole public suite, publish `scores.json` (must say `complete: true`), open a pull request on the Space or kit with the model repo on the Hub, the engine commit, hardware and settings. The maintainers then run the private parts themselves, so the weights must be public or shared with them.

## The kit

`pip install -e .` from a checkout. `suite rebuild` builds the files from pinned public sources (not redistributed; about 7 GB download and 17 GB working space; the HLE terms must be accepted on the Hub first). `run` takes an engine: `--engine module:Class` for a subclass of `decision_index.engines.Engine` with `__call__(state, questions) -> (response, raw)`; `Unsupported` marks a declared capacity limit. `score` computes every benchmark and the index. Our adapter is [D29](/design/d29-decision-index-adapter.md).

## What it means for us

- Only the public 20% can be scored by us; the private 80% needs a submission.
- Many public benchmarks (MMLU, ARC, HellaSwag, WinoGrande, ANLI, GSM8K, BANKING77, CLINC150) are tasks that tasksource or FLAN contain, so Mimir has very likely seen them and our mixture may contain their train splits. Our leakage guards compare train rows with our own evaluation rows, not with the index rows; an overlap check is the first step of [E15](/experiments/e15-decision-index-public-suite.md).
- Several benchmarks are far from our training data (tool use, retrieval with 32 candidates, chess, music, long-context claims). Prompts over our 2,048-token training length count as unsupported unless `max_len` is raised.
