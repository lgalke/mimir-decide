---
type: Template
title: 'Template: experiment'
description: Copy to experiments/eNN-slug.md to plan or record an experiment.
tags: [template]
timestamp: '2026-10-04T00:00:00Z'
---

# Template: experiment

```yaml
---
type: Experiment
title: "ENN: title"
description: one sentence
status: planned | running | done | abandoned
answers: [qNN]
depends_on: [eNN]
timestamp: YYYY-MM-DDT00:00:00Z
---
```

Sections: Hypothesis, Prerequisites, Setup (exact commands), Metrics to record, Decision rule, Status and results (table). Write the decision rule **before** running. Never evaluate on the test split to choose between options.
