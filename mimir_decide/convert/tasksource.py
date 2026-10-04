"""tasksource/tasksource-jev-typed-decisions -> Decision.

Schema (verified 2026-10-04): state, kind (choice|score|noul), id, question, options, target, source,
variant, split, group_id, question_id, license, license_use. Noul rows have empty `options` and a one-element
`target` holding P(true). Splits are the dataset's own train/validation/test; we never move rows between them.
"""
from __future__ import annotations

from typing import Iterator, Optional

from .common import Counter0, make_decision, norm_split

REPO = "tasksource/tasksource-jev-typed-decisions"
SPLIT_MAP = {"train": "train", "validation": "validation", "test": "test"}


def row_to_decision(r: dict, split: str, drops: Counter0):
    if norm_split(r.get("split")) not in (None, split):
        drops.inc("tasksource_split_column_mismatch")
        return None
    kind = r["kind"]
    opts, tgt = list(r.get("options") or []), list(r.get("target") or [])
    values = None
    if kind == "noul":
        if len(tgt) != 1:
            drops.inc("noul_bad_target")
            return None
        p = min(1.0, max(0.0, float(tgt[0])))
        opts, tgt = ["yes", "no"], [p, 1.0 - p]
    elif kind in ("choice", "score"):
        if kind == "score":
            values = [float(i) for i in range(len(opts))]
    else:
        drops.inc("unknown_kind")
        return None
    d = make_decision(
        id=r["id"], case_id=r["group_id"], group_id=r["group_id"], dataset="tasksource",
        source=f"tasksource/{r['source']}", split=split, language="und",
        license=r.get("license") or "", kind=kind, state=r["state"], question=r["question"],
        options=opts, target=tgt, option_values=values, variant=r.get("variant"),
    )
    if d is None:
        drops.inc("empty_target")
    return d


def iter_records(split: str, drops: Counter0, limit: Optional[int] = None,
                 revision: Optional[str] = None, streaming: bool = True) -> Iterator:
    from datasets import load_dataset

    ds = load_dataset(REPO, "default", split=SPLIT_MAP[split], streaming=streaming, revision=revision)
    n = 0
    for r in ds:
        d = row_to_decision(r, split, drops)
        if d is None:
            continue
        # license_use is carried in the license string so downstream policy sees it
        if r.get("license_use") == "non-commercial" and "nc" not in d.license.lower():
            d.license = f"{d.license} (non-commercial)"
        yield d
        n += 1
        if limit is not None and n >= limit:
            return
