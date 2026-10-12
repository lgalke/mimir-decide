---
type: Observation
title: 'O21: Our benchmark scorer reproduces the published Uniform row for KL and Brier, not for accuracy and ECE'
description: The Typed Decisions README names its metrics but does not define them. Testing definitions against the published Uniform and Prior rows pinned down KL and Brier, and left accuracy and ECE open.
date: '2026-10-12'
confidence: high
tags: [e14, benchmark, scorer]
timestamp: 2026-10-12T00:00:00Z
---

## Observation

Source: the 2,000 test decisions and 6,000 train decisions of `LocalLLaMA/typed-decisions`, scored with `mimir_decide.typed_decisions --references`.

| Reference | Metric | This scorer | Published |
|---|---|---|---|
| Uniform | Accuracy | 0.278 | 0.308 |
| Uniform | KL from gold | **0.444** | **0.444** |
| Uniform | Brier | **0.238** | **0.238** |
| Uniform | ECE | 0.046 | 0.169 |
| Prior (mean gold per workflow and question) | Accuracy | 0.481 | 0.470 |
| Prior | KL from gold | 0.327 | 0.347 |
| Prior | Brier | 0.181 | 0.189 |
| Prior | ECE | 0.036 | 0.088 |

Variants tried without success for Uniform accuracy: first or last tie-breaking (0.278, 0.304), expected accuracy over ties (0.318), the dataset's option order for yes/no questions, the hard `label` field instead of the argmax. Variants for ECE: 10, 15 and 20 bins against hard or soft correctness, unbinned and class-wise forms. Variants for the Prior: mean gold, argmax label frequencies (KL 0.439), pooling across workflows.

## Interpretation

- **KL and Brier are pinned down:** KL(gold‖prediction) in natural log and the sum over options of the squared difference to the gold distribution, averaged over decisions. This matches the benchmark for the Uniform row to three decimals.
- **Accuracy and ECE are not pinned down.** The README itself says that ECE 'is computed differently by different submitters' and that KL and Brier are 'the columns to read, not ECE'. Our accuracy is the agreement of the most probable option with the gold's.
- **The Prior row** cannot be reproduced from the description ('each question's `train` label frequencies'); our version (mean gold per workflow and question) is close but not equal.
- **Consequence:** numbers from our scorer are comparable with other numbers from our scorer, and with the published KL and Brier of the Uniform-type rows. For the comparison with Anna's approach, the same scorer should be applied to both sets of predictions; the saved per-decision probabilities allow that.

## Reproduce

`python -m mimir_decide.typed_decisions --references --final`

## Related

* [E14](/experiments/e14-typed-decisions-benchmark.md)
* [D28](/design/d28-typed-decisions-scorer-and-test-use.md)
