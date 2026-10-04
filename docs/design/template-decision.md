---
type: Template
title: 'Template: decision record'
description: Copy to design/dNN-slug.md to record a design decision.
tags: [template]
timestamp: '2026-10-04T00:00:00Z'
---

# Template: decision record

Copy this file, set `type: Decision`, give it the next id, then regenerate indexes (`python -m mimir_decide.okf --write`) and add a line to `docs/log.md`.

```yaml
---
type: Decision
title: "DNN: short title"
description: one sentence
status: proposed | accepted | superseded
date: YYYY-MM-DD
decided_by: user | agent | both
tags: []
timestamp: YYYY-MM-DDT00:00:00Z
---
```

Sections: Context, Decision, Consequences, Revisit when, Related. Link affected pages with absolute links such as `[D05](/design/d05-slot-readout-heads.md)`. If it supersedes another decision, set that one to `status: superseded` and link both ways. Never change the meaning of an accepted decision silently.
