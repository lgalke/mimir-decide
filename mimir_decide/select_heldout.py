"""Select held-out SOURCES whose label sets are not seen in training (experiment E12).

    python -m mimir_decide.select_heldout --data_dir $DATA --seed 0 --out configs/heldout_unseen_labels.yaml

A source qualifies when its option sets look like a fixed label set that no other training source uses:
  repeat_fraction  >= 0.5   share of its rows whose exact option set occurs at least `--min_repeat` times in the source
                            (excludes per-item multiple-choice QA, where every row has different answer options)
  seen_fraction    <= 0.1   share of its rows whose exact option set also occurs in the TRAIN rows of another source
  string_overlap   <= 0.3   share of its distinct option strings that also occur in the TRAIN options of another source
  n_train >= 100 and n_eval >= 150 (validation + calibration + test rows)
Option strings are compared after normalisation (NFKC, lower case, whitespace collapsed). Selection is seeded and
stratified by the dominant kind. Sources that Mimir's training mix does not contain by name (reports/mimir_overlap.json)
are picked separately (`--n_unflagged`): they give a test that is unseen for both our fine-tuning and Mimir.
Commit the output file BEFORE any model is trained on the derived mixture (pre-registration).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from .schema import norm_text

FILES = {"train": "train.parquet", "validation": "validation.parquet", "calib": "calib.parquet",
         "test": "eval/test/test.parquet"}


def source_stats(data_dir: Path) -> dict:
    """Per source: kinds, n_train, n_eval, Counter of option sets (all rows), Counter of option sets (train rows)."""
    st: dict = defaultdict(lambda: {"kinds": Counter(), "n_train": 0, "n_eval": 0, "sets_all": Counter(),
                                    "sets_train": Counter(), "k": []})
    for split, rel in FILES.items():
        p = data_dir / rel
        if not p.exists():
            continue
        for r in pq.read_table(p, columns=["source", "kind", "options"]).to_pylist():
            s = st[r["source"]]
            fs = frozenset(norm_text(o) for o in r["options"])
            s["kinds"][r["kind"]] += 1
            s["sets_all"][fs] += 1
            s["k"].append(len(fs))
            if split == "train":
                s["n_train"] += 1
                s["sets_train"][fs] += 1
            else:
                s["n_eval"] += 1
    return st


def score_sources(st: dict, min_repeat: int = 20) -> dict:
    set_sources: dict = defaultdict(set)     # option set -> sources that use it in TRAIN rows
    string_sources: dict = defaultdict(set)  # option string -> sources that use it in TRAIN rows
    for name, s in st.items():
        for fs in s["sets_train"]:
            set_sources[fs].add(name)
            for o in fs:
                string_sources[o].add(name)
    out = {}
    for name, s in st.items():
        total = sum(s["sets_all"].values())
        repeat = sum(c for c in s["sets_all"].values() if c >= min_repeat) / total
        seen = sum(c for fs, c in s["sets_all"].items() if set_sources.get(fs, set()) - {name}) / total
        strings = {o for fs in s["sets_all"] for o in fs}
        overlap = sum(1 for o in strings if string_sources.get(o, set()) - {name}) / max(1, len(strings))
        ks = sorted(s["k"])
        out[name] = {"kind": s["kinds"].most_common(1)[0][0], "n_train": s["n_train"], "n_eval": s["n_eval"],
                     "median_options": ks[len(ks) // 2], "repeat_fraction": round(repeat, 3),
                     "seen_fraction": round(seen, 3), "string_overlap": round(overlap, 3),
                     "label_set": sorted(next(iter(s["sets_all"].most_common(1)))[0])[:8]}
    return out


def eligible(scores: dict, min_train=100, min_eval=150, max_seen=0.1, max_overlap=0.3, min_repeat_fraction=0.5):
    ok, why = [], Counter()
    for name, m in scores.items():
        if m["n_train"] < min_train: why["n_train too small"] += 1
        elif m["n_eval"] < min_eval: why["n_eval too small"] += 1
        elif m["repeat_fraction"] < min_repeat_fraction: why["no fixed label set (per-item options)"] += 1
        elif m["seen_fraction"] > max_seen: why["option set used by another source"] += 1
        elif m["string_overlap"] > max_overlap: why["option strings used by other sources"] += 1
        else: ok.append(name)
    return sorted(ok), why


def pick(names: list[str], scores: dict, n: int, rng: random.Random) -> list[str]:
    """Seeded pick of n names, round-robin over the dominant kinds."""
    by_kind: dict[str, list[str]] = defaultdict(list)
    for x in names:
        by_kind[scores[x]["kind"]].append(x)
    for v in by_kind.values():
        rng.shuffle(v)
    chosen, kinds = [], sorted(by_kind)
    while len(chosen) < n and any(by_kind[k] for k in kinds):
        for k in kinds:
            if by_kind[k] and len(chosen) < n:
                chosen.append(by_kind[k].pop())
    return sorted(chosen)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_flagged", type=int, default=8, help="sources that Mimir's mix contains by name")
    ap.add_argument("--n_unflagged", type=int, default=2, help="sources Mimir's mix does not contain by name")
    ap.add_argument("--min_repeat", type=int, default=20)
    ap.add_argument("--min_train", type=int, default=100)
    ap.add_argument("--min_eval", type=int, default=150)
    ap.add_argument("--max_seen", type=float, default=0.1, help="max share of rows whose option set another source uses")
    ap.add_argument("--max_overlap", type=float, default=0.3, help="max share of option strings another source uses")
    ap.add_argument("--out", required=True, help="YAML file with held_out_sources (commit it before training)")
    a = ap.parse_args(argv)
    data = Path(a.data_dir).expanduser()
    scores = score_sources(source_stats(data), a.min_repeat)
    ok, why = eligible(scores, a.min_train, a.min_eval, a.max_seen, a.max_overlap)
    ov = data / "reports" / "mimir_overlap.json"
    seen_by_mimir = set(json.loads(ov.read_text())["sources_seen_by_mimir"]) if ov.exists() else None
    if seen_by_mimir is None:
        print("WARNING: reports/mimir_overlap.json not found; all sources are treated as flagged")
        flagged, unflagged = ok, []
    else:
        flagged = [x for x in ok if x in seen_by_mimir]
        unflagged = [x for x in ok if x not in seen_by_mimir]
    rng = random.Random(a.seed)
    chosen_un = pick(unflagged, scores, a.n_unflagged, rng)
    chosen_fl = pick(flagged, scores, a.n_flagged, rng)
    chosen = sorted(chosen_un + chosen_fl)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# Held-out sources with label sets unseen in training (E12). Generated by mimir_decide.select_heldout,\n"
              f"# seed {a.seed}, {len(scores)} sources scored, {len(ok)} eligible ({len(flagged)} flagged / {len(unflagged)} not flagged by name).\n"
              f"# Not flagged: {', '.join(chosen_un) or '(none)'}\n")
    out.write_text(header + yaml.safe_dump({"held_out_sources": chosen}, sort_keys=False))
    rep = {"seed": a.seed, "n_scored": len(scores), "n_eligible": len(ok), "rejected": dict(why),
           "chosen": {x: {**scores[x], "flagged_by_name": (x in seen_by_mimir) if seen_by_mimir is not None else None}
                      for x in chosen}}
    out.with_suffix(".report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("n_scored", "n_eligible", "rejected")}, indent=1))
    for x in chosen:
        m = scores[x]
        print(f"  {x:<55} {m['kind']:<7} train={m['n_train']:<5} eval={m['n_eval']:<5} K~{m['median_options']:<3} "
              f"seen={m['seen_fraction']} overlap={m['string_overlap']} {'(not flagged)' if x in chosen_un else ''}")
    if len(chosen_un) < a.n_unflagged or len(chosen_fl) < a.n_flagged:
        print(f"WARNING: fewer sources than requested (unflagged {len(chosen_un)}/{a.n_unflagged}, "
              f"flagged {len(chosen_fl)}/{a.n_flagged}); loosen the thresholds or lower the counts, and log it")
    print(f"wrote {out} and {out.with_suffix('.report.json')}")


if __name__ == "__main__":
    main()
