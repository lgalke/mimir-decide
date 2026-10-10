"""Build the evaluation-only set of extra tasks for E12 (option A of D26).

    python -m mimir_decide.build_extra_eval --pilot_dir $DATA --out_dir $DATA_EXTRA \
        --sources_out configs/extra_eval_sources.yaml

Candidates are sources that are NOT in the training mixture (`<pilot_dir>/train.parquet`):
  * bekko subsets whose upstream licences pass the allowlist but whose review status is 'qualified' (and verified
    ones that were not selected), validation and test splits;
  * allowlisted tasksource sources absent from the mixture, validation and test splits.
Unknown, non-commercial and not-allowlisted licences stay excluded. The rows are used for EVALUATION ONLY.

Each candidate row must pass, against the pilot's TRAIN rows: the exact state hash and the n-gram fingerprints (the same
guards as the mixture builder). Each source must then pass the label-novelty rule (a fixed label set that the training
sources do not use): repeat_fraction >= 0.5, seen_fraction <= 0.1, string_overlap <= 0.3, and at least `--min_eval` rows.

Outputs: <out_dir>/extra_eval/extra_eval.parquet (evaluate with `--split extra`), <out_dir>/reports/extra_eval_report.json
(every candidate with licence, row counts, drops and novelty numbers), <out_dir>/mixture_manifest.json, and the source
list (`held_out_sources`, the key `compare --sources-file` reads). Commit the source list before evaluating any model on it.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
import yaml

from . import licenses
from .build_mixture import cap_options
from .convert import bekko, tasksource
from .convert.common import Counter0
from .schema import Decision, fingerprints, state_hash, write_parquet, norm_text


def train_vocab(pilot_dir: Path):
    """Option sets, option strings, source names and state hashes of the pilot's TRAIN rows."""
    t = pq.read_table(pilot_dir / "train.parquet", columns=["source", "options", "state"])
    sets, strings, sources, states = set(), set(), set(), []
    for r in t.to_pylist():
        fs = frozenset(norm_text(o) for o in r["options"])
        sets.add(fs)
        strings |= fs
        sources.add(r["source"])
        states.append(r["state"])
    return sets, strings, sources, states


def candidate_rows(cfg: dict, drops: Counter0, excluded: Counter0, accept_qualified: bool = True,
                   limit: int | None = None) -> Iterable[Decision]:
    """Validation and test rows of bekko and tasksource. Licence-failing rows are dropped by the caller."""
    s = cfg["sources"]
    manifests = bekko.load_manifests(s["bekko"].get("revision"))
    for split in ("validation", "test"):
        yield from bekko.iter_records(split, drops, limit, s["bekko"].get("revision"), accept_qualified,
                                      manifests=manifests, excluded=excluded)
    for split in ("validation", "test"):
        yield from tasksource.iter_records(split, drops, limit, s["tasksource"].get("revision"), True)


def _squash(name: str) -> list[str]:
    """Lower-case alphanumeric path components of a source name without its dataset prefix."""
    import re

    return [re.sub(r"[^a-z0-9]+", "", part.lower()) for part in name.split("/")[1:] if part]


def familiar_task(name: str, train_sources: set) -> str | None:
    """Name of a training source that carries the same task (by name), else None.

    A candidate counts as familiar when a path component of its name equals, or (for 5+ characters) contains or is
    contained in, a component of a training source's name; for example bekko/snli and tasksource/snli."""
    mine = [c for c in _squash(name) if len(c) >= 3]
    for t in sorted(train_sources):
        for c in _squash(t):
            for m in mine:
                if m == c or (len(m) >= 5 and len(c) >= 5 and (m in c or c in m)):
                    return t
    return None


def score_candidates(rows_by_source: dict[str, list[Decision]], sets: set, strings: set, min_repeat: int = 20) -> dict:
    out = {}
    for name, rows in rows_by_source.items():
        fsets = [frozenset(norm_text(o) for o in d.options) for d in rows]
        counts = Counter(fsets)
        n = len(rows)
        own_strings = {o for fs in fsets for o in fs}
        out[name] = {
            "n_rows": n,
            "repeat_fraction": round(sum(c for c in counts.values() if c >= min_repeat) / n, 3),
            "seen_fraction": round(sum(1 for fs in fsets if fs in sets) / n, 3),
            "string_overlap": round(sum(1 for o in own_strings if o in strings) / max(1, len(own_strings)), 3),
            "median_options": sorted(len(fs) for fs in fsets)[n // 2],
            "kind": Counter(d.kind for d in rows).most_common(1)[0][0],
            "label_set": sorted(counts.most_common(1)[0][0])[:8],
        }
    return out


def leak_filter(rows: list[Decision], train_states: list[str], boiler_df: int = 3):
    """Drop rows whose state occurs in the train rows (exact) or shares n-gram fingerprints with them."""
    from .schema import text_hash

    train_hashes = {text_hash(s) for s in train_states}
    keep, dropped = [], Counter()
    survivors = []
    for d in rows:
        if state_hash(d) in train_hashes:
            dropped["leak_state_in_train"] += 1
        else:
            survivors.append(d)
    # candidate fingerprints, ignoring template text that many candidate states share
    df: dict[int, int] = defaultdict(int)
    seen_states = set()
    row_fps = {}
    for i, d in enumerate(survivors):
        fps = fingerprints(d.state)
        row_fps[i] = fps
        h = state_hash(d)
        if h not in seen_states:
            seen_states.add(h)
            for fp in fps:
                df[fp] += 1
    cand = {fp for fp, c in df.items() if c <= boiler_df}
    matched = set()
    done = set()
    for s in train_states:
        h = text_hash(s)
        if h in done:
            continue
        done.add(h)
        for fp in fingerprints(s):
            if fp in cand:
                matched.add(fp)
    for i, d in enumerate(survivors):
        fps = row_fps[i]
        hits = sum(1 for fp in fps if fp in matched)
        if hits >= 2 or (hits >= 1 and len(fps) <= 2):
            dropped["leak_ngram_in_train"] += 1
        else:
            keep.append(d)
    return keep, dropped


def build(cfg: dict, pilot_dir: Path, out_dir: Path, sources_out: Path, rows: Iterable[Decision] | None = None,
          min_eval: int = 150, max_per_source: int = 400, seed: int = 0, max_state_chars: int = 12000,
          max_options: int = 26, max_seen: float = 0.1, max_overlap: float = 0.3, min_repeat_fraction: float = 0.5) -> dict:
    pilot_dir, out_dir = Path(pilot_dir).expanduser(), Path(out_dir).expanduser()
    drops, excluded = Counter0(), Counter0()
    sets, strings, train_sources, train_states = train_vocab(pilot_dir)
    rows = candidate_rows(cfg, drops, excluded) if rows is None else rows

    why, by_source = Counter(), defaultdict(list)
    lic = {}
    for d in rows:
        ok, reason = licenses.check(d.license)
        if not ok:
            why[f"licence:{reason}"] += 1
            continue
        if d.source in train_sources:
            why["source_in_training_mixture"] += 1
            continue
        if d.validate() or len(d.state) > max_state_chars or cap_options(d, max_options) is None:
            why["invalid_or_too_long"] += 1
            continue
        lic[d.source] = d.license
        by_source[d.source].append(d)

    rng = random.Random(seed)
    for name, lst in by_source.items():          # cap per source before the expensive leakage pass
        if len(lst) > max_per_source:
            rng.shuffle(lst)
            by_source[name] = lst[:max_per_source]
    flat = [d for lst in by_source.values() for d in lst]
    kept, leak_drops = leak_filter(flat, train_states)
    why.update(leak_drops)
    kept_by_source: dict[str, list[Decision]] = defaultdict(list)
    for d in kept:
        kept_by_source[d.source].append(d)

    scores = score_candidates(kept_by_source, sets, strings)
    chosen, rejected = [], {}
    for name, m in sorted(scores.items()):
        if m["n_rows"] < min_eval: rejected[name] = "fewer eval rows than --min_eval after the leakage filter"
        elif m["repeat_fraction"] < min_repeat_fraction: rejected[name] = "no fixed label set (per-item options)"
        elif m["seen_fraction"] > max_seen: rejected[name] = "option set used by a training source"
        elif m["string_overlap"] > max_overlap: rejected[name] = "option strings used by training sources"
        else: chosen.append(name)

    familiar = {n: familiar_task(n, train_sources) for n in chosen}
    task_novel = [n for n in chosen if familiar[n] is None]
    final = [d for n in chosen for d in kept_by_source[n]]
    (out_dir / "extra_eval").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    write_parquet(final, out_dir / "extra_eval" / "extra_eval.parquet")
    report = {"purpose": "evaluation only (D26 option A); never train, calibrate or select on these rows",
              "n_sources_with_rows": len(by_source), "n_sources_chosen": len(chosen), "rows": len(final),
              "row_drops": dict(why), "bekko_subsets_excluded_by_licence": dict(excluded),
              "n_task_novel_sources": len(task_novel),
              "chosen": {n: {**scores[n], "license": lic[n], "familiar_task_in_training": familiar[n]} for n in chosen},
              "rejected": {n: {"reason": r, **scores[n]} for n, r in rejected.items()}}
    (out_dir / "reports" / "extra_eval_report.json").write_text(json.dumps(report, indent=1))
    (out_dir / "mixture_manifest.json").write_text(json.dumps(
        {"counts": {"extra_eval": len(final)}, "derived": {"kind": "extra_eval", "pilot_dir": str(pilot_dir),
                                                           "sources": chosen, "task_novel_sources": task_novel}}, indent=1))
    sources_out = Path(sources_out)
    sources_out.parent.mkdir(parents=True, exist_ok=True)
    sources_out.write_text(f"# Extra evaluation sources for E12 (evaluation only; D26 option A). Generated by build_extra_eval.\n"
                           f"# {len(chosen)} sources, {len(final)} rows; {len(task_novel)} of them have no same-name task in training.\n"
                           f"# held_out_sources = new label sets (all chosen); task_novel_sources = the subset whose task is also new.\n"
                           f"# Commit before evaluating any model on them.\n"
                           + yaml.safe_dump({"held_out_sources": chosen, "task_novel_sources": task_novel}, sort_keys=False))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot_dir", required=True, help="mixture the models were trained on (train.parquet)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sources_out", required=True, help="YAML source list to commit before evaluation")
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--min_eval", type=int, default=150)
    ap.add_argument("--max_per_source", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_seen", type=float, default=0.1)
    ap.add_argument("--max_overlap", type=float, default=0.3)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config))
    rep = build(cfg, a.pilot_dir, a.out_dir, a.sources_out, min_eval=a.min_eval, max_per_source=a.max_per_source,
                seed=a.seed, max_seen=a.max_seen, max_overlap=a.max_overlap)
    print(json.dumps({k: rep[k] for k in ("n_sources_with_rows", "n_sources_chosen", "rows", "row_drops")}, indent=1))
    for n, m in rep["chosen"].items():
        fam = m["familiar_task_in_training"]
        print(f"  {n:<55} {m['kind']:<7} rows={m['n_rows']:<4} K~{m['median_options']:<3} seen={m['seen_fraction']} "
              f"overlap={m['string_overlap']} lic={m['license']}" + (f"  [familiar task: {fam}]" if fam else ""))
    print(f"{rep['n_task_novel_sources']} of {rep['n_sources_chosen']} chosen sources have no same-name task in training")
    print(f"{len(rep['rejected'])} candidate sources rejected (see reports/extra_eval_report.json)")


if __name__ == "__main__":
    main()
