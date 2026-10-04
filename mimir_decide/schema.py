"""Unified decision record shared by all converters, the mixture builder and training."""
from __future__ import annotations

import hashlib
import math
import zlib
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Optional

import pyarrow as pa

KINDS = ("noul", "choice", "score")
SPLITS = ("train", "validation", "test")


@dataclass
class Decision:
    id: str
    case_id: str
    group_id: str
    dataset: str          # collection, e.g. "tasksource", "massive"
    source: str           # task inside the collection, e.g. "tasksource/boolq"
    split: str
    language: str
    license: str
    kind: str             # noul | choice | score
    state: str
    question: str
    options: list[str]
    target: list[float]   # distribution over options, sums to 1
    option_values: Optional[list[float]] = None  # numeric value per option (score)
    soft: bool = False    # True when target is a non-one-hot distribution
    variant: Optional[str] = None

    def validate(self) -> Optional[str]:
        """Return None when valid, otherwise a short drop reason."""
        if self.kind not in KINDS:
            return "bad_kind"
        if self.split not in SPLITS:
            return "bad_split"
        if len(self.options) < 2:
            return "lt2_options"
        if len(self.options) != len(self.target):
            return "target_len_mismatch"
        if any(not math.isfinite(t) or t < 0 for t in self.target):
            return "bad_target_value"
        if abs(sum(self.target) - 1.0) > 1e-4:
            return "target_not_normalised"
        if self.option_values is not None and len(self.option_values) != len(self.options):
            return "option_values_mismatch"
        if not self.state.strip() or not self.question.strip():
            return "empty_text"
        if any(not o.strip() for o in self.options):
            return "empty_option"
        return None

    def to_row(self) -> dict:
        return asdict(self)


def normalise_target(target: list[float]) -> Optional[list[float]]:
    """Clip to >= 0 and renormalise; None when there is no mass."""
    t = [max(0.0, float(x)) for x in target]
    s = sum(t)
    if not math.isfinite(s) or s <= 0:
        return None
    return [x / s for x in t]


def is_soft(target: list[float]) -> bool:
    return max(target) < 1.0 - 1e-6


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip().lower()


def text_hash(s: str) -> str:
    return hashlib.blake2b(norm_text(s).encode("utf-8"), digest_size=12).hexdigest()


def state_hash(d: Decision) -> str:
    return text_hash(d.state)


def query_hash(d: Decision) -> str:
    return text_hash(d.state + "\x1f" + d.question)


def full_hash(d: Decision) -> str:
    return text_hash(d.state + "\x1f" + d.question + "\x1f" + "\x1e".join(d.options))


def fingerprints(text: str, n: int = 12, mod: int = 4) -> list[int]:
    """Rendering-independent content fingerprints: crc32 of every word n-gram whose hash % mod == 0.

    Catches the same underlying example rendered differently by two collections (e.g. HelpSteer2 appears both as
    our `helpsteer2` converter and inside tasksource). States with < 2n words get no fingerprints (they rely on
    the exact state hash and group ids)."""
    w = norm_text(text).split()
    if len(w) < 2 * n:
        return []
    out = set()
    for i in range(len(w) - n + 1):
        h = zlib.crc32(" ".join(w[i : i + n]).encode("utf-8"))
        if h % mod == 0:
            out.add(h)
    return sorted(out)


def bucket(key: str, seed: int = 0) -> float:
    """Deterministic value in [0,1) from a string key (for group-wise splits)."""
    h = hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2**64


ARROW_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("case_id", pa.string()),
        ("group_id", pa.string()),
        ("dataset", pa.string()),
        ("source", pa.string()),
        ("split", pa.string()),
        ("language", pa.string()),
        ("license", pa.string()),
        ("kind", pa.string()),
        ("state", pa.large_string()),
        ("question", pa.string()),
        ("options", pa.list_(pa.string())),
        ("target", pa.list_(pa.float64())),
        ("option_values", pa.list_(pa.float64())),
        ("soft", pa.bool_()),
        ("variant", pa.string()),
    ]
)


def write_parquet(records: list[Decision], path) -> None:
    import pyarrow.parquet as pq

    rows = [r.to_row() for r in records]
    table = pa.Table.from_pylist(rows, schema=ARROW_SCHEMA)
    pq.write_table(table, path, compression="zstd")


def read_parquet(path) -> list[Decision]:
    import pyarrow.parquet as pq

    out = []
    for row in pq.read_table(path).to_pylist():
        out.append(Decision(**row))
    return out
