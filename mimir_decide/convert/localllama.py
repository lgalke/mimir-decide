"""LocalLLaMA/typed-decisions -> Decision (Apache-2.0).

Verified columns (2026-10-04): id, workflow, split, state (JSON str), questions (JSON str), gold (JSON str),
factors, label_agreement, n_questions. `questions[q] = {type, instructions, criteria}` where criteria is a
dict id->description (choice/noul) or an ordered list of descriptions (score); `gold[q].probabilities` maps the
criterion id (or score level "0".."3") to a probability. Upstream splits are train (1,200 rows over 4 workflows)
and test (400). We carve our validation from train group-wise by row id (10%); upstream test stays test.
"""
from __future__ import annotations

import json
from typing import Iterator, Optional

from ..schema import bucket
from .common import Counter0, make_decision, norm_split, render_state

REPO = "LocalLLaMA/typed-decisions"
LICENSE = "apache-2.0"
VAL_FRACTION = 0.1


def row_to_decisions(r: dict, split: str, drops: Counter0) -> list:
    rid = r["id"]
    state = render_state(json.loads(r["state"]))
    questions, gold = json.loads(r["questions"]), json.loads(r["gold"])
    source = f"localllama/{r.get('workflow', 'unknown')}"
    out = []
    for qname, q in questions.items():
        g = gold.get(qname)
        if g is None or "probabilities" not in g:
            drops.inc("localllama_missing_gold")
            continue
        probs, kind, crit = g["probabilities"], q["type"], q.get("criteria")
        values = None
        if kind == "score" and crit:
            n = len(crit)
            opts = [str(c) for c in crit]
            tgt = [float(probs.get(str(i), 0.0)) for i in range(n)]
            values = [float(i) for i in range(n)]
        elif kind == "noul":
            opts = [crit["true"], crit["false"]] if crit else ["yes", "no"]
            tgt = [float(probs["true"]), float(probs["false"])]
        elif kind == "choice" and crit:
            keys = list(crit)
            opts = [f"{k}: {crit[k]}" for k in keys]
            tgt = [float(probs.get(k, 0.0)) for k in keys]
        else:
            drops.inc(f"localllama_unsupported_type:{kind}")
            continue
        d = make_decision(
            id=f"{rid}:{qname}", case_id=rid, group_id=rid, dataset="localllama", source=source, split=split,
            language="en", license=LICENSE, kind=kind, state=state, question=q["instructions"], options=opts,
            target=tgt, option_values=values,
        )
        if d:
            out.append(d)
        else:
            drops.inc("localllama_empty_target")
    return out


def iter_records(split: str, drops: Counter0, limit: Optional[int] = None,
                 revision: Optional[str] = None, **_) -> Iterator:
    """`split` is OUR split. validation is carved from upstream train; test is upstream test."""
    from datasets import load_dataset

    upstream = "test" if split == "test" else "train"
    ds = load_dataset(REPO, "all", split=upstream, revision=revision)
    n = 0
    for r in ds:
        if norm_split(r.get("split")) not in (None, upstream):  # type: ignore[union-attr]
            drops.inc("localllama_split_column_mismatch")
            continue
        if upstream == "train":
            is_val = bucket(r["id"], seed=17) < VAL_FRACTION  # type: ignore[index]
            if (split == "validation") != is_val:
                continue
        for d in row_to_decisions(r, split, drops):  # type: ignore[arg-type]
            yield d
            n += 1
            if limit is not None and n >= limit:
                return
