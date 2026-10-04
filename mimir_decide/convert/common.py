"""Helpers shared by converters."""
from __future__ import annotations

import json
from typing import Iterator, Optional

from ..schema import Decision, is_soft, normalise_target


_SPLIT_ALIASES = {"dev": "validation", "val": "validation", "valid": "validation", "eval": "validation"}


def norm_split(x):
    """Map upstream split labels (dev/val/...) to train|validation|test; None stays None."""
    return None if x is None else _SPLIT_ALIASES.get(str(x).lower(), str(x).lower())


def jloads(s, default=None):
    """Parse a JSON-encoded string field; return the raw string if it is not JSON."""
    if s is None:
        return default
    if not isinstance(s, str):
        return s
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return s


def render_state(obj) -> str:
    """Render a JSON state object as readable text (key: value per line)."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                v = json.dumps(v, ensure_ascii=False)
            parts.append(f"{k}: {v}")
        return "\n".join(parts)
    return json.dumps(obj, ensure_ascii=False)


def make_decision(
    *,
    id: str,
    case_id: str,
    group_id: str,
    dataset: str,
    source: str,
    split: str,
    language: str,
    license: str,
    kind: str,
    state: str,
    question: str,
    options: list[str],
    target: list[float],
    option_values: Optional[list[float]] = None,
    variant: Optional[str] = None,
) -> Optional[Decision]:
    t = normalise_target(target)
    if t is None:
        return None
    return Decision(
        id=id, case_id=case_id, group_id=group_id, dataset=dataset, source=source, split=split,
        language=language, license=license, kind=kind, state=state, question=question,
        options=[str(o) for o in options], target=t, option_values=option_values,
        soft=is_soft(t), variant=variant,
    )


class Counter0(dict):
    def inc(self, key: str, n: int = 1) -> None:
        self[key] = self.get(key, 0) + n


def limited(it: Iterator, limit: Optional[int]) -> Iterator:
    if limit is None:
        yield from it
        return
    for i, x in enumerate(it):
        if i >= limit:
            return
        yield x
