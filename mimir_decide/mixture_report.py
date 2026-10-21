"""Diversity report of a mixture (D31): what is in a training set, in numbers that can be compared between mixtures.

    python -m mimir_decide.mixture_report --data_dir $DATA/mixture-v1 [--file train_v1-371k.parquet] [--compare $DATA/mixture-v0]

Reported: decisions, sources, effective number of sources (exp of the entropy of the source shares), share of the
largest source and of the 10 largest, shares per family (dataset prefix), kind shares, option-count histogram,
mean target entropy and mean majority-label share, length quantiles in characters, and a script mix (share of states by
dominant writing system: a rough proxy, not language identification).
"""
from __future__ import annotations

import argparse
import json
import math
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

from .schema import read_parquet

SCRIPTS = ("LATIN", "CYRILLIC", "CJK", "HIRAGANA", "KATAKANA", "HANGUL", "ARABIC", "DEVANAGARI", "GREEK", "HEBREW", "THAI")


def dominant_script(text: str, limit: int = 400) -> str:
    c = Counter()
    for ch in text[:limit]:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            c[next((s for s in SCRIPTS if name.startswith(s)), "OTHER")] += 1
    return c.most_common(1)[0][0] if c else "NONE"


def entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def report(decs) -> dict:
    n = len(decs)
    src = Counter(d.source for d in decs)
    shares = np.array(sorted(src.values(), reverse=True), dtype=float) / max(n, 1)
    fam = Counter(d.source.split("/")[1] if d.source.startswith("tasksource/procedural") else d.dataset for d in decs)
    kinds = Counter(d.kind for d in decs)
    ks = Counter(min(len(d.options), 27) for d in decs)
    tent = np.array([entropy_bits(np.array(d.target)) for d in decs]) if n else np.zeros(1)
    maj = np.array([max(d.target) for d in decs]) if n else np.zeros(1)
    lens = np.array([len(d.state) + sum(len(o) for o in d.options) for d in decs]) if n else np.zeros(1)
    scripts = Counter(dominant_script(d.state) for d in decs)
    return {
        "decisions": n, "sources": len(src),
        "effective_sources": round(float(math.exp(-(shares[shares > 0] * np.log(shares[shares > 0])).sum())), 1) if n else 0,
        "top1_share": round(float(shares[0]), 4) if n else 0, "top10_share": round(float(shares[:10].sum()), 4) if n else 0,
        "sources_with_at_least_1000": sum(1 for v in src.values() if v >= 1000),
        "by_dataset": {k: round(v / n, 4) for k, v in fam.most_common()},
        "kind_share": {k: round(kinds[k] / n, 4) for k in ("choice", "noul", "score")},
        "binary_share": round(ks[2] / n, 4), "options_histogram": {str(k): ks[k] for k in sorted(ks)},
        "mean_target_entropy_bits": round(float(tent.mean()), 3), "mean_top_label_mass": round(float(maj.mean()), 3),
        "soft_share": round(sum(d.soft for d in decs) / n, 4) if n else 0,
        "chars_p50_p90_p99": [int(np.percentile(lens, q)) for q in (50, 90, 99)],
        "script_share": {k: round(v / n, 4) for k, v in scripts.most_common(6)},
        "largest_sources": src.most_common(8),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--file", default="train.parquet")
    ap.add_argument("--compare", default=None, help="another mixture directory (its train.parquet) to print next to this one")
    ap.add_argument("--out", default=None, help="write JSON here (default: reports/diversity_<file>.json in data_dir)")
    a = ap.parse_args(argv)
    d = Path(a.data_dir).expanduser()
    rep = report(read_parquet(d / a.file))
    out = Path(a.out) if a.out else d / "reports" / f"diversity_{Path(a.file).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    other = report(read_parquet(Path(a.compare).expanduser() / "train.parquet")) if a.compare else None
    keys = ["decisions", "sources", "effective_sources", "top1_share", "top10_share", "sources_with_at_least_1000",
            "binary_share", "mean_target_entropy_bits", "mean_top_label_mass", "soft_share", "chars_p50_p90_p99"]
    print(f"{'':<28}{a.file:<28}" + (f"{'compare':<28}" if other else ""))
    for k in keys:
        print(f"{k:<28}{str(rep[k]):<28}" + (f"{str(other[k]):<28}" if other else ""))
    for k in ("kind_share", "by_dataset", "script_share"):
        print(f"{k}: {rep[k]}" + (f"\n  compare: {other[k]}" if other else ""))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
