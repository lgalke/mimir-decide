"""Tables over Decision Index runs (E15): the index and area scores, per-benchmark skill, coverage and length statistics.

    python -m mimir_decide.di_table runs/di-baseline-v0 runs/di-clef-probe-v1 --labels baseline-v0 clef-probe
    python -m mimir_decide.di_table runs/di-baseline-v0 --lengths          # statuses and token lengths from results.jsonl(.gz)

Reads the kit's `scores.json` (index, raw_index, areas, benchmarks) tolerantly; a missing key prints '-'.
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


def by_id(x) -> dict:
    """The kit stores areas as a list of {id, ...}; older shapes are a dict keyed by id."""
    if isinstance(x, dict):
        return {str(k): v for k, v in x.items()}
    if isinstance(x, list):
        return {str(v.get("id", v.get("name", i))): v for i, v in enumerate(x) if isinstance(v, dict)}
    return {}


def load_scores(run: Path) -> dict:
    s = json.loads((run / "scores.json").read_text())
    s["areas"] = by_id(s.get("areas"))
    if not s.get("benchmarks"):
        bs = run / "benchmark-summary.json"
        if bs.exists():
            s["benchmarks"] = by_id(json.loads(bs.read_text()).get("benchmarks", json.loads(bs.read_text())))
    s["benchmarks"] = by_id(s.get("benchmarks"))
    return s


def shape(x, depth=0, maxdepth=3):
    """A short description of a JSON value, to find the right keys of the kit's files."""
    if isinstance(x, dict):
        return {k: (shape(v, depth + 1, maxdepth) if depth < maxdepth else type(v).__name__) for k, v in list(x.items())[:12]}
    if isinstance(x, list):
        return [len(x), shape(x[0], depth + 1, maxdepth)] if x else []
    return x if isinstance(x, (int, float, str, bool)) or x is None else type(x).__name__


def fmt(x, nd=1):
    if isinstance(x, dict):
        return json.dumps(x, separators=(",", ":"))[:60]
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}" if isinstance(x, float) and x <= 1.0 else fmt(x)


def summary_table(runs: list[Path], labels: list[str]) -> str:
    scores = [load_scores(r) for r in runs]
    areas = sorted({a for s in scores for a in (s.get("areas") or {})})
    head = ["model", "index", "raw index"] + [f"{a} skill" for a in areas] + ["coverage %"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for lab, s in zip(labels, scores):
        ar = s.get("areas") or {}
        cov = [v.get("coverage") for v in ar.values() if isinstance(v, dict) and v.get("coverage") is not None]
        row = [lab, fmt(s.get("index", s.get("decision_index"))), fmt(s.get("raw_index", s.get("raw")))] + [fmt((ar.get(a) or {}).get("skill")) for a in areas]
        row.append(pct(sum(cov) / len(cov)) if cov else "-")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def benchmark_table(runs: list[Path], labels: list[str]) -> str:
    scores = [load_scores(r) for r in runs]
    names = sorted({str(k) for s in scores for k in (s.get("benchmarks") or {})}, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0, x))
    lines = ["| benchmark | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for n in names:
        cells = []
        for s in scores:
            b = (s.get("benchmarks") or {}).get(n) or {}
            cells.append(f"{fmt(b.get('skill'))} ({pct(b.get('coverage'))}%)" if b else "-")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def length_stats(results: Path, bins=(512, 1024, 2048, 4096, 8192)) -> dict:
    """Status counts and, from the engine's raw output, the number of requests above each token length."""
    p = results / "results.jsonl.gz" if (results / "results.jsonl.gz").exists() else results / "results.jsonl"
    opener = gzip.open if str(p).endswith(".gz") else open
    status, by_bin, per_ds = Counter(), Counter(), defaultdict(Counter)
    n = 0
    with opener(p, "rt") as f:
        for line in f:
            r = json.loads(line)
            n += 1
            status[r.get("status")] += 1
            per_ds[r.get("dataset")][r.get("status")] += 1
            t = (r.get("raw_output") or {}).get("max_tokens") if isinstance(r.get("raw_output"), dict) else None
            if t is not None:
                for b in bins:
                    by_bin[b] += t > b
    return {"requests": n, "status": dict(status), "longer_than": {str(b): by_bin[b] for b in bins},
            "unsupported_by_benchmark": {d: c["unsupported"] for d, c in per_ds.items() if c.get("unsupported")}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--benchmarks", action="store_true", help="also print the per-benchmark table")
    ap.add_argument("--keys", action="store_true", help="print the structure of scores.json and benchmark-summary.json of the first run")
    ap.add_argument("--lengths", action="store_true", help="print status and length statistics of each run instead")
    a = ap.parse_args(argv)
    runs = [Path(r) for r in a.runs]
    labels = a.labels or [r.name for r in runs]
    if a.keys:
        for f in ("scores.json", "benchmark-summary.json"):
            print(f, json.dumps(shape(json.loads((runs[0] / f).read_text())), indent=1)[:3500])
        return
    if a.lengths:
        for lab, r in zip(labels, runs):
            print(lab, json.dumps(length_stats(r)))
        return
    print(summary_table(runs, labels))
    if a.benchmarks:
        print("\n" + benchmark_table(runs, labels))


if __name__ == "__main__":
    main()
