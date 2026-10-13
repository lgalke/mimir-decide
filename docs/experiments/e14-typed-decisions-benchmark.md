---
type: Experiment
title: 'E14: Typed Decisions benchmark (comparison with Anna''s approach)'
description: Score our models on the full test split of LocalLLaMA/typed-decisions (400 cases, 2,000 decisions) with accuracy, KL from gold, Brier and ECE, to compare with a second approach on the same data.
status: done
answers: [q04]
depends_on: [e10, e11]
tags: [benchmark, comparison, typed-decisions, priority]
timestamp: 2026-10-12T00:00:00Z
---

## Purpose

Top priority from the owner (2026-10-12): a PhD student works on a Mimir-Decide approach of her own and uses the test split of [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions). Our models are scored on the same data, so that the two approaches can be compared.

## What the benchmark is

A synthetic benchmark of four workflows (agent traces, customer service, invoices, security incidents). The test split has 400 cases with 5 typed questions each: 2,000 decisions (600 noul, 600 choice, 800 score). Gold is the mean of three teacher samples, so it is a soft distribution. Accuracy is agreement with the teacher: the README gives 0.735 as the teacher's own agreement (a ceiling) and 0.470 as the prior. The README keeps two tables that are not comparable: general models scored zero-shot, and models fitted on `train`.

## Where our models belong

- **Untrained Mimir with the letter prompt** (`letter-zeroshot`): general, zero-shot. It has never seen these workflows.
- **Our fine-tuned models** (`slot-v0`, `baseline-v0`, `baseline-lmhead`, later the probes): trained on a broad mixture that **includes this benchmark's train split** (by design about 90% of its 6,000 decisions; the other 10% is our validation set; the exact number is in the server's `mixture_manifest.json`) among about 330 other sources. The test split was never used for training or calibration. They belong to the second table, 'trained on `train` and other data', like the Bekko rows.
- Anna's approach: she knows what it was trained on. The tables must not be mixed.

## Mode

Every decision is answered on its own: one forward pass per question, with the case's state and that question's options. The README's reference rows send the whole case with all five questions in one request, and warns that the mode matters (Jev's yes/no accuracy was 0.843 with one question per request and 0.788 with five). Our mode is the 'one question per request' mode.

## Scorer

`python -m mimir_decide.typed_decisions` (definitions in its docstring). KL from gold is KL(gold‖prediction) in natural log; Brier is the sum over options of the squared difference to the gold distribution; both reproduce the published **Uniform** row exactly (KL 0.444, Brier 0.238). Accuracy is agreement of the most probable option with the gold's; ECE uses 15 bins on the top-1 confidence. The README does not define these two, and they could not be reproduced (Uniform accuracy is 0.278 with this scorer against 0.308 published; the Prior row differs too, see [O21](/observations/o21-benchmark-scorer-matches-the-published-uniform-row-only-for-kl-and-brier.md)). **Compare approaches only with the same scorer.** Each run therefore also writes the per-decision probabilities (`*_predictions.json`: case and question id, option keys, probabilities), so that Anna's predictions can be scored with the identical code, or ours with hers.

## Setup

Run on the server from the repository root. Each model is scored **once**: one invocation reads the test split, computes the logits once, and writes both the calibrated and the raw version (the access is logged in `eval/test_access_log.jsonl`).

```bash
cd /work/dfm/lukasgp/mimir-decide
git pull
source .env.sh

# 1. The reference rows (no model): Uniform and Prior
python -m mimir_decide.typed_decisions --references --final

# 2. Untrained Mimir with the letter prompt (general, zero-shot)
python -m mimir_decide.typed_decisions --run_dir runs/letter-zeroshot --checkpoint zero-shot --final

# 3. The trained models
python -m mimir_decide.typed_decisions --run_dir runs/slot-v0 --final
python -m mimir_decide.typed_decisions --run_dir runs/baseline-v0 --final
python -m mimir_decide.typed_decisions --run_dir runs/baseline-lmhead --final

# 4. Commit the small result files (check git status first: only runs/*/eval/typed_decisions_test*.json)
git add runs/letter-zeroshot/eval runs/slot-v0/eval runs/baseline-v0/eval runs/baseline-lmhead/eval
git commit -m "Add Typed Decisions benchmark results"
git push
```

Another approach (for example Anna's) is scored with the same code from its predictions. The file is JSON, either a list of entries or a dict from id to entry. The id is `<case id>:<question>` (the dataset row `id`, a colon, the question name). An entry has `probabilities` as `{option key: probability}`, or `keys` and `probs` as in our own files. Option keys are the criterion ids for choice questions, `"0"`...`"n-1"` for scores, and `true`/`false` (or `yes`/`no`) for noul. The order does not matter, and each distribution is renormalised. All 2,000 decisions must be present.

```bash
python -m mimir_decide.typed_decisions --score_predictions path/to/anna_predictions.json --name anna --final
```

The result goes to `runs/external-anna/eval/typed_decisions_test.json`. Rescoring our own `slot-v0` predictions this way reproduces its own numbers exactly.

The probes of [E13](/experiments/e13-head-only-baselines.md) are scored the same way (`--run_dir runs/letter-probe --final`, `--run_dir runs/slot-probe --final`). Both are done: letter probe accuracy 0.611 / KL 0.301 / Brier 0.164; slot probe 0.504 / 0.369 / 0.201 ([O23](/observations/o23-letter-probe-recovers-a-fifth-of-the-fine-tuning-gain.md), [O24](/observations/o24-slot-probe-extracts-little-from-a-frozen-backbone.md)).

## Metrics to record

Per model, calibrated and raw: accuracy, KL from gold, Brier, ECE (and NLL), overall and per kind, per workflow and per question. Compare with the README's rows with the reservations above, and with Anna's numbers rescored with this scorer.

## Rules

No model or setting is chosen with the test split. Each model is scored once. The test split is used for reporting only; the owner approved this use because the comparison is the priority. No result is submitted to the benchmark's discussion board without the owner's decision.

## Limits

One seed per model. Our models answer each question separately. The scorer's accuracy and ECE definitions are our reading of the benchmark, not its official ones. The gold measures agreement with a teacher: scores above 0.735 mean learning the teacher's quirks, not being more correct.

## Status and results

Status: **done** for the four models of the first batch (the probes of E13 and Anna's approach are still to be scored). Results are in [O22](/observations/o22-typed-decisions-benchmark-results.md).

| Date | Run id / path | Config (overrides) | Result | Observation page |
|---|---|---|---|---|
| 2026-10-12 | `runs/letter-zeroshot`, `slot-v0`, `baseline-v0`, `baseline-lmhead` | one pass per model, calibrated and raw | accuracy 0.607 / 0.788 / 0.788 / 0.779; KL 0.295 / 0.086 / 0.087 / 0.086; Brier 0.166 / 0.048 / 0.048 / 0.048 | [O22](/observations/o22-typed-decisions-benchmark-results.md) |
