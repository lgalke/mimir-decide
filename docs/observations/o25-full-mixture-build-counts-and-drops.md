---
type: Observation
title: 'O25: Full mixture build, counts and drops'
description: The pilot mixture has 371,275 train decisions. Licence policy removes about 456k tasksource rows, n-gram guards about 27k. LocalLLaMA lost 2,066 rows, so less of its train split is in the mixture than planned.
date: '2026-10-14'
confidence: medium
tags: [e01, mixture, drops, leakage]
timestamp: 2026-10-14T00:00:00Z
---

## Observation

Source: `mixture_manifest.json` of `mixture-v0` on the server, pasted by the owner (keys `counts`, `drops`, `heldout_patterns_unmatched`). The keys `train_by_kind`, `train_by_dataset` and `train_sources` were not pasted.

| Split | Decisions |
|---|---|
| train | 371,275 |
| validation | 6,127 |
| calib | 6,189 |
| heldout (massive/nb-NO) | 600 |
| test | 11,674 |

`heldout_patterns_unmatched` is empty: every held-out pattern matched a source.

| Collection | Drops (rows) |
|---|---|
| tasksource | licence unknown 304,065; non-commercial 103,214; not on allowlist 48,508; n-gram leak 21,204; state too long 5,058; empty text 4,864; duplicate 3,344; state hash leak 46 |
| bekko | read limit per source 1,914,885; n-gram leak 2,676; state too long 1,577; state hash leak 553; duplicate 120 |
| localllama | duplicate 1,606; n-gram leak 460 |
| massive | held-out task 11,514; state hash leak 992 |
| helpsteer2 | n-gram leak 2,517; group leak 580; duplicate 30; state hash leak 5 |

## Interpretation

- **E01 rule:** train 371,275 is above the 300k threshold and `heldout_patterns_unmatched` is empty. The third condition (manual inspection of 20 `ngram_leak_examples.jsonl` rows) is not done.
- **Licences dominate the loss in tasksource:** 455,787 rows (88% of its recorded drops) fall to the licence policy ([D02](/design/d02-permissive-licences-only.md)). Unknown licences alone are 304k. Admitting them would more than double the mixture, but that is the owner's decision (Q06).
- **Guards:** 26,857 rows fell to n-gram leaks and 1,596 to exact state hashes across all sources; no guard drop looks like a bug by size, but the n-gram rate for helpsteer2 (2,517) deserves the manual check.
- **bekko `read_limit_per_source` (1.9M) is a read cap, not a quality loss;** the per-source cap of the mixture builder applies after it.
- **Correction for E14:** LocalLLaMA lost 2,066 rows (1,606 duplicates, 460 n-gram) from a train split of about 6,000 decisions of which about 90% were planned. So at most about 3,300 (56%) of the benchmark's train decisions are in the mixture, not 90%. The exact number needs `train_by_dataset` or `train_sources`. The duplicates are probably identical state, question and options across cases of a synthetic dataset; this was not checked. The 'trained on train and other data' label of E14 still holds, with a smaller share.
- **Test rows:** 11,674 test decisions exist in the mixture's `eval/test`; these are the upstream tests (HelpSteer2 validation, MASSIVE, tasksource, bekko).

## Reproduce

On the server: `python -c "import json,os; m=json.load(open(os.environ['MIMIR_DECIDE_DATA']+'/mixture-v0/mixture_manifest.json')); print(json.dumps({k:m[k] for k in ('counts','drops','heldout_patterns_unmatched')},indent=1))"`.

## Related

* [E01](/experiments/e01-full-mixture-build.md)
* [E14](/experiments/e14-typed-decisions-benchmark.md)
* [D12](/design/d12-ngram-fingerprint-guard-cross-source-only.md)
