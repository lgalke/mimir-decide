"""nvidia/HelpSteer2 -> Decision (CC-BY-4.0).

Upstream has only train (20,324) and validation (1,038). Our mapping:
  upstream train      -> our train (95%, group-wise by prompt) and our validation (5%, group-wise)
  upstream validation -> our TEST (never used for training/selection; upstream validation is the standard
                         reward-model benchmark split, so keeping it as test makes numbers comparable)
Each rating attribute (0-4) becomes one Score decision with hard targets.
"""
from __future__ import annotations

from typing import Iterator, Optional

from ..schema import bucket
from .common import Counter0, make_decision

REPO = "nvidia/HelpSteer2"
LICENSE = "cc-by-4.0"
VAL_FRACTION = 0.05

ATTRIBUTES = {
    "helpfulness": "How helpful is the response to the prompt?",
    "correctness": "How factually correct and complete is the response?",
    "coherence": "How coherent and clear is the response?",
    "complexity": "How complex is the writing (0 = basic language, 4 = expert-level)?",
    "verbosity": "How verbose is the response relative to what the prompt asks for (0 = very terse, 4 = very verbose)?",
}
LEVELS = ["0 (lowest)", "1", "2", "3", "4 (highest)"]


def row_to_decisions(r: dict, idx: int, split: str, drops: Counter0, upstream: str = "train") -> list:
    prompt = r["prompt"]
    group = f"helpsteer2:{prompt[:200]}"
    state = f"Prompt:\n{prompt}\n\nResponse:\n{r['response']}"
    out = []
    for attr, q in ATTRIBUTES.items():
        v = r.get(attr)
        if v is None or int(v) not in range(5):
            drops.inc("helpsteer_bad_rating")
            continue
        tgt = [0.0] * 5
        tgt[int(v)] = 1.0
        d = make_decision(
            id=f"helpsteer2:{upstream}:{idx}:{attr}", case_id=f"helpsteer2:{upstream}:{idx}", group_id=group, dataset="helpsteer2",
            source=f"helpsteer2/{attr}", split=split, language="en", license=LICENSE, kind="score",
            state=state, question=q, options=LEVELS, target=tgt, option_values=[0, 1, 2, 3, 4],
        )
        if d:
            out.append(d)
    return out


def iter_records(split: str, drops: Counter0, limit: Optional[int] = None,
                 revision: Optional[str] = None, streaming: bool = True) -> Iterator:
    from datasets import load_dataset

    upstream = "validation" if split == "test" else "train"
    ds = load_dataset(REPO, split=upstream, revision=revision)
    n = 0
    for i, r in enumerate(ds):
        if upstream == "train":
            is_val = bucket(f"helpsteer2:{r['prompt'][:200]}", seed=23) < VAL_FRACTION
            if (split == "validation") != is_val:
                continue
        for d in row_to_decisions(r, i, split, drops, upstream):
            yield d
            n += 1
            if limit is not None and n >= limit:
                return
