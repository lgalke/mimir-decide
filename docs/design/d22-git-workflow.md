---
type: Decision
title: 'D22: Git workflow for agents'
description: Agents work on cluster/* branches with frequent commits; the owner merges; pushing needs permission; data and checkpoints never enter the repository.
status: proposed
date: '2026-10-04'
decided_by: owner initialised the repository; workflow proposed by agent, pending owner confirmation
tags: [process, git, handoff]
timestamp: '2026-10-04T00:00:00Z'
---

## Context

The owner initialised git after the handoff request. `main` holds one commit, `ad89493` ("first commit"): 121 tracked files (code, configs, docs, `uv.lock`), no data or checkpoints, remote `origin` = `git@github.com:lgalke/mimir-decide.git`, tracking `origin/main`. An agent on a GPU cluster will continue the work and produces results, new docs and possibly code changes that the owner has to be able to review.

## Decision

Section 9 of `AGENTS.md` is the operative text. In short:

- Agents work on `cluster/<topic>` branches, never directly on `main`; the owner merges.
- Small commits that bundle a change with its tests, docs and log line; one commit per recorded experiment; messages explain why and cite experiment and decision ids.
- Before committing: offline tests, `python -m mimir_decide.okf --check`, and a look at `git status`.
- Data, mixtures, caches, checkpoints, run directories, environments, secrets and files over about 5 MB are never committed; results enter the repository as short numbers in experiment and observation pages.
- Local commits need no permission. Pushing a `cluster/*` branch needs the owner's yes; pushing to `main`, force-pushes, history rewrites, tags, releases and pull requests need an explicit request.
- Generated indexes are regenerated, not hand-merged; log bullets from both sides are kept; ids stay unique and sequential.
- A cluster-specific `uv.lock` change is not committed without the owner's agreement.

## Consequences

- Every experiment leaves a reviewable trail (commit, log entry, observation page).
- Nothing large or sensitive can slip into history by default, because `.gitignore` plus the 'never commit' list cover the usual suspects. Anything committed to a pushed history is hard to remove, which is why pushing is gated.
- The rule 'ask the owner before pushing or publishing' in `AGENTS.md` section 4 is unchanged; section 9 spells out what that means for branches.

## Revisit when

The owner prefers a pull-request flow, a different branch naming, or allows agents to push by default; or the repository starts to need large-file tooling (for example Git LFS for small reference artifacts).

## Related

* [D21](/design/d21-docs-as-okf-bundle.md)
* [D11](/design/d11-split-isolation-and-test-access.md)
