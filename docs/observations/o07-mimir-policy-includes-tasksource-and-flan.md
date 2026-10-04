---
type: Observation
title: 'O07: Mimir v1.5 trained on tasksource and FLAN-style data'
description: The sampling policy lists tasksource__, flan__, flan_factual__, posttrain_natural_instructions__ and sapient-synth-flan-*.
date: '2026-10-04'
confidence: high
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Observation

`dfm11_sampling_policy.yaml` has 193 prefixes. Relevant ones: `tasksource__`, `flan__`, `flan__cot_`, `flan_factual__`, `posttrain_natural_instructions__`, `sapient-synth-flan-*`, `allenai_tulu_*`, `giannor_dala_*`, `giannor_gec_*`, `alexandra_dane__`, `ai_arena*`. HelpSteer2 and MASSIVE are not listed by name.

## Interpretation

Results on tasksource-derived data may reflect familiarity; the DFM10 base data is private, so only names can be compared.

## Reproduce

`python -m mimir_decide.audit_mimir_overlap --data_dir <mixture>`.

## Related

* [mimir-v1-5](/research/mimir-v1-5.md)
* [q01-clean-generalisation-evidence](/questions/q01-clean-generalisation-evidence.md)
