"""Overlap of Decision Index requests with our training mixture (E15, step 1).

    python -m mimir_decide.di_overlap --rows work/suite/selected-rows.jsonl.gz --train $DATA/train.parquet

For every request the text `state + instructions` of each question is compared with the training states: an exact
match of the normalised text (as in the leakage guards) or at least two shared word 12-gram fingerprints
(`schema.fingerprints`, the same rule as the builder: two hits, or one hit when the text has at most two fingerprints).
The report counts requests per benchmark. It only flags; nothing is dropped. The Decision Index puts the passage of
many tasks into the instructions and leaves the state empty, so the instructions are part of the compared text.
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from .schema import fingerprints, text_hash


def render(x) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, sort_keys=True)


def train_index(train_file: Path, limit: int | None = None):
    """(set of exact state hashes, set of fingerprints) of the training states."""
    hashes, fps = set(), set()
    pf = pq.ParquetFile(train_file)
    n = 0
    for batch in pf.iter_batches(columns=["state"], batch_size=20000):
        for s in batch.column("state").to_pylist():
            hashes.add(text_hash(s))
            fps.update(fingerprints(s))
            n += 1
            if limit and n >= limit:
                return hashes, fps
    return hashes, fps


def request_texts(row: dict) -> list[str]:
    st = render(row.get("state", ""))
    out = []
    for q in (row.get("questions") or {}).values():
        ins = render(q.get("instructions", ""))
        out.append((st + "\n" + ins).strip())
    return out or [st]


def flag(text: str, hashes: set, fps: set) -> tuple[bool, bool]:
    """(exact match, n-gram match) of one text against the training index."""
    exact = text_hash(text) in hashes
    mine = fingerprints(text)
    hits = [f for f in mine if f in fps]
    ngram = len(hits) >= 2 or (bool(hits) and len(mine) <= 2)
    return exact, ngram


def scan(rows_path: Path, hashes: set, fps: set, limit: int | None = None) -> dict:
    opener = gzip.open if str(rows_path).endswith(".gz") else open
    stats = defaultdict(Counter)
    with opener(rows_path, "rt") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            row = json.loads(line)
            ds = (row.get("_evaluation") or {}).get("dataset") or row.get("family") or "unknown"
            flags = [flag(t, hashes, fps) for t in request_texts(row)]
            c = stats[ds]
            c["requests"] += 1
            c["exact"] += any(e for e, _ in flags)
            c["ngram"] += any(g for _, g in flags)
            c["either"] += any(e or g for e, g in flags)
    return {k: dict(v) for k, v in stats.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True, help="suite rows file (jsonl or jsonl.gz)")
    ap.add_argument("--train", required=True, help="train.parquet of the mixture")
    ap.add_argument("--out", default="di_overlap.json")
    ap.add_argument("--limit_train", type=int, default=None, help="smoke tests: only the first N training states")
    ap.add_argument("--limit_rows", type=int, default=None)
    a = ap.parse_args(argv)
    hashes, fps = train_index(Path(a.train), a.limit_train)
    print(f"training index: {len(hashes):,} states, {len(fps):,} fingerprints")
    stats = scan(Path(a.rows), hashes, fps, a.limit_rows)
    Path(a.out).write_text(json.dumps(stats, indent=1))
    print(f"{'benchmark':<34}{'requests':>9}{'exact':>8}{'n-gram':>8}{'either %':>10}")
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]["either"] / max(kv[1]["requests"], 1)):
        print(f"{k[:33]:<34}{v['requests']:>9}{v['exact']:>8}{v['ngram']:>8}{100 * v['either'] / v['requests']:>9.1f}%")
    tot = sum(v["requests"] for v in stats.values())
    print(f"total requests {tot}, flagged {sum(v['either'] for v in stats.values())}; wrote {a.out}")


if __name__ == "__main__":
    main()
