---
type: Question
title: 'Q06: How much of bekko survives the licence policy?'
description: Most subsets are 'qualified' and excluded; the full-build size is unknown.
status: open
priority: high
resolved_by: [e01]
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Why it matters

bekko is the largest candidate (6.59M cases).

## What we know

Trial build: of 17 licence-excluded subsets seen, 12 were 'qualified', 2 unknown, 3 not on the allowlist ([D13](/design/d13-bekko-licence-and-test-handling.md)); only 7 bekko sources entered the trial mixture.

## How to resolve

Full build ([E01](/experiments/e01-full-mixture-build.md)); compare decisions and diversity with and without `accept_qualified`; review which qualified caveats are acceptable.

## Decision impact

Decides whether the mixture reaches the colleague's 500k-1M pilot size.
