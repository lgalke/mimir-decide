---
type: Decision
title: 'D32: Licence parser may ignore provenance annotations (opt-in, train pass only)'
description: Annotations such as '(original source)' and '(publisher)' after a permissive licence name no longer cause exclusion when license_strip_annotations is on. About 90k tasksource rows gain. Everything else stays as in D02.
status: accepted
date: '2026-10-21'
decided_by: owner (approved the fix); implementation by agent
tags: [licences, policy, data]
timestamp: 2026-10-21T00:00:00Z
---

## Context

[D02](/design/d02-permissive-licences-only.md) keeps only permissive licences and treats a compound string as allowed only if every part is allowed. tasksource marks some licences with provenance notes, for example `cc-by-4.0, cc-by-4.0 (original source)`. The parser read the note as a part of the name and excluded the rows. In the `full` config this affects 89,995 train rows (tasksource `sources.yaml`, 2026-10-21): 28,909 `cc-by-4.0 ... (original source)`, 21,646 `mit ...`, 11,258 `apache-2.0 ...`, 10,652 `cc-by-sa-4.0 ...`, 8,145 `apache-2.0 (original source)`, and smaller groups.

## Decision

- `licenses.check(lic, use, strip_annotations=False)`: with `True`, the annotations `(original source)` and `(publisher)` after a licence name are ignored. Any other text next to a licence still fails the allowlist, for example `mit (publisher); original review terms (original source)` (9,310 rows stay excluded), `Custom (DPI)`, `gpl-3.0`, `odbl` and `other, cdla-permissive-1.0`. Non-commercial markers still win.
- **Opt-in.** The default is `False`, so the v0 mixture, its eval files and the pre-registered E12 candidate list are reproduced exactly. `configs/data_v1.yaml` sets `license_strip_annotations: true` for the **train pass only**; the eval pass keeps the strict parser, so the frozen evaluation does not change.
- The build writes `reports/license_newly_admitted.csv` (source, licence, rows) so the owner can review what the change admits, and the manifest counts it.

## Consequences

- About 90k more rows are available to v1 (before the guards and caps).
- Share-alike handling is unchanged: CC-BY-SA rows are listed in the manifest (`share_alike_train_sources`).
- Admitting more licence families (unknown, custom, GPL) would be a separate owner decision; it is not part of this one.

## Revisit when

The owner reviews `license_newly_admitted.csv` and objects to a source, or tasksource changes its annotation style.

## Related

* [D02](/design/d02-permissive-licences-only.md)
* [D31](/design/d31-mixture-v1-size-and-diversity.md)
