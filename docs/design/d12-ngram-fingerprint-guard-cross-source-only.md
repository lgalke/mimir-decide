---
type: Decision
title: 'D12: N-gram fingerprint guard compares against other sources only'
description: Rendering-independent content check, applied across sources but not within one.
status: accepted
date: '2026-10-04'
decided_by: agent
tags: [leakage]
timestamp: '2026-10-04T00:00:00Z'
---

## Context

Exact hashes miss the same example rendered differently (tasksource contains HelpSteer2, cladder, math_qa that bekko also contains). The first version also compared within a source and dropped 1,345 LocalLLaMA decisions, because synthetic templated data repeats scenario text across its own train and test ([O06](/observations/o06-localllama-template-overlap-false-positives.md)).

## Decision

Word 12-grams, sampled by `crc32 % 4 == 0`, ignored if shared by more than 3 eval states (template text); a train row is dropped on 2 or more hits against eval rows of other sources (or 1 hit if it has at most 2 fingerprints). Examples are logged to `reports/ngram_leak_examples.jsonl`.

## Consequences

- Catches real overlaps ([O04](/observations/o04-helpsteer2-upstream-train-validation-overlap.md), [O05](/observations/o05-cross-collection-duplicates.md)).
- Within a source the upstream split is trusted. States under 24 words get no fingerprints and rely on exact hashes.
- Not paraphrase-proof.

## Revisit when

A new templated source shows mass drops, or paraphrased duplicates are suspected (try minhash).

## Related

* [dataset-selection](/design/dataset-selection.md)
