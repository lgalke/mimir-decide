---
type: Decision
title: Dataset selection, licence policy and leakage controls
description: Which sources feed Mimir-Decide, how each is split, what is excluded and why, and how train/eval separation is enforced.
status: accepted
date: '2026-10-04'
decided_by: user and agent
tags: [datasets, licences, splits, leakage, evaluation]
timestamp: 2026-10-04T00:00:00Z
---

# Dataset selection, licence policy and leakage controls

Follows [colleague-recommendations](/research/colleague-recommendations.md); decisions by the project owner on 2026-10-04: skip in-house data for now, **permissive licences only**, single GPU node.

## Sources and split mapping

| Source | Licence | Train | Validation | Test | Notes |
|---|---|---|---|---|---|
| tasksource-jev-typed-decisions, config `default` | per row | train | validation | test | Upstream `split` column says `dev` for validation; labels are normalised. Noul rows have empty options and a single target = P(true); we render them as yes/no. |
| bekko-system-one-dataset-v0 | per upstream dataset | manifest train subsets | manifest `evaluation` entries with split=validation | entries with split=test, **hashed only, not written** | Membership from `training-manifest.json`; quarantine list is empty today. Ranking decisions become Choice over documents. |
| LocalLLaMA/typed-decisions | Apache-2.0 | 90% of upstream train, by row id | other 10% | upstream test | Real columns are JSON strings (`state`, `questions`, `gold`); the dataset card summary was misleading. |
| MASSIVE 1.1 (da, en, sv, nb, de) | CC-BY-4.0 | train | dev | test | HF repo is a loading script, so the source archive is read directly. Eight candidate intents per utterance (gold + sampled), humanised names. |
| HelpSteer2 | CC-BY-4.0 | 95% of upstream train, by prompt | other 5% | upstream validation | Upstream has no test split; its validation split becomes our test. Five attributes become five Score decisions. |

## Licence policy (permissive only)

Allowed: Apache-2.0, MIT, BSD, ISC, CC0, CC-BY (2.0/3.0/4.0), CC-BY-SA (3.0/4.0), ODC-BY, PDDL, Unlicense, AFL-3.0. Everything else is excluded: unknown or "unspecified", "other", non-commercial, or not on the list. A compound licence string passes only if every part passes. Excluded sources are listed with row counts in `reports/license_exclusions.csv`. The policy removes a lot: of the first 4,000 tasksource train rows read in a trial build (ordered by source name, so not a random sample), 1,249 were licence-unknown, 397 non-commercial and 190 on neither list, i.e. 46% excluded.

- **bekko** licences are per upstream dataset in `sources.json`. A subset is used only if every upstream licence is permissive **and** its review status is "verified". Subsets marked "qualified" (a data-specific caveat) are excluded by default (`accept_qualified: false`); this is conservative and removes a large share of bekko.
- **CC-BY-SA** sources are allowed but listed in the manifest (`share_alike_train_sources`): whether share-alike reaches model weights is unsettled.
- **Excluded outright:** ChaosNLI (CC-BY-NC, and derived from SNLI/MNLI/αNLI dev sets); bekko's MS MARCO and GooAQ subsets (non-commercial).

## Leakage controls (details in [training-setup](/design/training-setup.md))

Held-out tasks: `massive/nb-NO` is removed from training entirely and evaluated separately (chosen before any training run).

1. Evaluation data of every source is read first, in every licence, and hashed: exact state hash, group id, and rendering-independent word 12-gram fingerprints. Train rows matching any of them are dropped.
2. **Why fingerprints:** the same example can reach us through two collections with different renderings, which an exact-hash guard cannot see. In a trial build (4,000 rows per source and split) the fingerprint guard caught real cases that the exact guard missed: `bekko/cladder` and `bekko/corr2cause` against their tasksource twins, `bekko/aqua_rat` against `tasksource/math_qa` (MathQA derives from AQuA), and HelpSteer2 rows whose response appears in both of the dataset's own upstream train and validation splits (5 of 1,038 validation responses, 2 prompts). Fingerprints ignore n-grams shared by many eval states (template text) and only compare against eval rows of **other sources**: within one source the upstream split is trusted, because templated synthetic data legitimately repeats scenario text across its own train and test (the first version compared within sources too and wrongly dropped 1,345 LocalLLaMA decisions).
3. Cross-collection dedup on state + question + options, preferring the first copy.
4. Final assertions on the written files fail the build on any residual overlap.

## Overlap with Mimir's own training

The v1.5 sampling policy lists `tasksource__`, `flan__`, `flan_factual__`, `posttrain_natural_instructions__` and `sapient-synth-flan-*` ([Mimir v1.5](/research/mimir-v1-5.md)). So Mimir has very likely already seen tasksource-derived and FLAN-style classification tasks, probably as generative instructions. The audit (`audit_mimir_overlap.py`) is **name-level only** because the DFM10 base data is not public. On the trial build it flagged:

| Collection | Flagged / checked sources | Why |
|---|---|---|
| tasksource | 316 / 316 | `tasksource__` is in the policy |
| bekko | 7 / 7 | built from public tasks covered by tasksource/FLAN/Natural-Instructions |
| HelpSteer2 | 5 / 5 | tasksource contains HelpSteer2 sources (alias rule) |
| MASSIVE | 2 / 2 | tasksource contains `multilingual/massive` (alias rule) |
| LocalLLaMA | 0 / 4 | no name match |

Consequences:

- Results on flagged sources measure the *format conversion and readout*, not generalisation to tasks Mimir has never seen. Almost everything we can download is flagged.
- **The held-out task `massive/nb-NO` is therefore not clean evidence** (I first assumed it was, and was wrong): tasksource's MASSIVE version may include Norwegian.
- Only LocalLLaMA is unflagged, and it is small, synthetic and templated. A clean generalisation test needs data Mimir cannot have seen, for example newly written or newly labelled Danish items; this is an open question ([open questions](/research/open-questions.md)).
- evaluate.py adds `sources_seen_by_mimir` to its output so the caveat sits next to every number.

Absence of a flag is not proof of no overlap, and a flag is not proof that specific eval rows were seen.
