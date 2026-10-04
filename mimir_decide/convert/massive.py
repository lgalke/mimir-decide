"""AmazonScience/massive (CC-BY-4.0) -> Decision.

The HF repo is a loading script (unsupported by recent `datasets`), so we read the official archive directly:
https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz
containing `1.1/data/<locale>.jsonl` with fields id, locale, partition (train|dev|test), scenario, intent, utt.

Each utterance becomes a Choice decision over `n_options` candidate intents (gold + random distractors, order
randomised, seeded by utterance id so train/eval construction is reproducible). Intent ids are humanised
(`alarm_set` -> `alarm: set`). Split mapping: train->train, dev->validation, test->test.
"""
from __future__ import annotations

import json
import os
import random
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterator, Optional

from .common import Counter0, make_decision

URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
LICENSE = "cc-by-4.0"
SPLIT_TO_PARTITION = {"train": "train", "validation": "dev", "test": "test"}
QUESTION = "Which intent does the user's utterance express?"


def humanise_intent(intent: str) -> str:
    scenario, _, rest = intent.partition("_")
    rest = rest.replace("_", " ")
    return f"{scenario}: {rest}" if rest else scenario


def fetch_archive(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / "amazon-massive-dataset-1.1.tar.gz"
    if not p.exists():
        tmp = cache_dir / f".massive-{os.getpid()}.tmp"  # per-process temp file: concurrent builds cannot collide
        urllib.request.urlretrieve(URL, tmp)
        os.replace(tmp, p)  # atomic; if another process won the race the result is identical
    return p


def read_locale(archive: Path, locale: str) -> list[dict]:
    rows = []
    with tarfile.open(archive, "r:gz") as tf:
        member = tf.getmember(f"1.1/data/{locale}.jsonl")
        f = tf.extractfile(member)
        assert f is not None
        for line in f:
            rows.append(json.loads(line))
    return rows


def rows_to_decisions(rows: list[dict], locale: str, split: str, all_intents: list[str], n_options: int,
                      seed: int, drops: Counter0) -> Iterator:
    part = SPLIT_TO_PARTITION[split]
    for r in rows:
        if r["partition"] != part:
            continue
        rng = random.Random(f"{seed}:{r['id']}:{locale}")
        gold = r["intent"]
        distract = [i for i in all_intents if i != gold]
        cands = rng.sample(distract, min(n_options - 1, len(distract))) + [gold]
        rng.shuffle(cands)
        tgt = [1.0 if c == gold else 0.0 for c in cands]
        d = make_decision(
            id=f"massive:{locale}:{r['id']}", case_id=f"massive:{locale}:{r['id']}",
            group_id=f"massive:{r['id']}",  # same utterance across locales shares a group
            dataset="massive", source=f"massive/{locale}", split=split, language=locale.split("-")[0],
            license=LICENSE, kind="choice", state=r["utt"], question=QUESTION,
            options=[humanise_intent(c) for c in cands], target=tgt,
        )
        if d:
            yield d
        else:
            drops.inc("massive_invalid")


def iter_records(split: str, drops: Counter0, limit: Optional[int] = None, locales=("da-DK", "en-US"),
                 n_options: int = 8, seed: int = 0, cache_dir: Optional[Path] = None) -> Iterator:
    archive = fetch_archive(Path(cache_dir or Path.home() / "mimir-decide-data" / "cache"))
    n = 0
    for loc in locales:
        rows = read_locale(archive, loc)
        all_intents = sorted({r["intent"] for r in rows})
        for d in rows_to_decisions(rows, loc, split, all_intents, n_options, seed, drops):
            yield d
            n += 1
            if limit is not None and n >= limit:
                return
