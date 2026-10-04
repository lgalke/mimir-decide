---
type: Decision
title: 'D21: Documentation is an OKF bundle with generated indexes'
description: docs/ follows the Open Knowledge Format; indexes are generated and checked; every decision, finding and result is logged.
status: accepted
date: '2026-10-04'
decided_by: user (requested) and agent
tags: [process, documentation]
timestamp: '2026-10-04T00:00:00Z'
---

## Context

The owner asked for the docs to be managed by OKF principles and for a handoff to another agent.

## Decision

`docs/` has research, design, questions, experiments and observations folders, plus `log.md`. Every page has frontmatter with `type`. Indexes come from `python -m mimir_decide.okf --write`; `--check` runs in the tests. Decisions are one page each (ids `dNN`); questions `qNN`; experiments `eNN`; observations `oNN`.

## Consequences

Any agent changing behaviour must add or update a page and a log line, then regenerate the indexes; see `AGENTS.md`.

## Revisit when

The structure becomes too heavy for the size of the project.

## Related

* [log](/log.md)
