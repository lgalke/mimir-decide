"""Build the Mimir-Decide mixture from the configured sources.

Order matters for leakage control:
  A. EVAL PASS   read validation + test of every source; hash ALL of it (every licence) into the leakage guard
                 and keep a capped, licence-allowed sample for the eval files.
  B. TRAIN PASS  read train splits only; drop rows that are invalid, too long, licence-excluded, from held-out
                 tasks, or that leak (state hash or group id seen in ANY eval split of ANY source); dedup; cap.
  C. WRITE       global temperature-based cap, parquet shards, manifest, reports, final zero-overlap assertion.

Outputs (under output_dir): train.parquet, validation.parquet, calib.parquet, heldout_tasks/*.parquet,
eval/test/*.parquet, mixture_manifest.json, reports/{license_exclusions.csv,drops.json}.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Optional

import yaml

from . import licenses
from .convert import bekko, helpsteer2, localllama, massive, tasksource
from .convert.common import Counter0
from .schema import Decision, bucket, fingerprints, full_hash, query_hash, state_hash, write_parquet


# ----------------------------------------------------------------------------- helpers
def cap_options(d: Decision, max_options: int) -> Optional[Decision]:
    """Reduce to <= max_options options: keep every option with target mass, fill with random others."""
    k = len(d.options)
    if k <= max_options:
        return d
    pos = [i for i, t in enumerate(d.target) if t > 0]
    if len(pos) > max_options:
        return None
    rng = random.Random(hashlib.blake2b(d.id.encode(), digest_size=8).digest())
    rest = [i for i in range(k) if d.target[i] == 0]
    keep = sorted(pos + rng.sample(rest, max_options - len(pos)))
    d.options = [d.options[i] for i in keep]
    d.target = [d.target[i] for i in keep]
    if d.option_values is not None:
        d.option_values = [d.option_values[i] for i in keep]
    return d


def is_heldout(source: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(source, p) for p in patterns)


class Reservoir:
    """Per-key reservoir sampling with a seeded RNG."""

    def __init__(self, cap: int, seed: int):
        self.cap, self.rng = cap, random.Random(seed)
        self.items: dict[str, list] = defaultdict(list)
        self.seen: dict[str, int] = defaultdict(int)

    def add(self, key: str, item) -> None:
        self.seen[key] += 1
        buf = self.items[key]
        if len(buf) < self.cap:
            buf.append(item)
        else:
            j = self.rng.randrange(self.seen[key])
            if j < self.cap:
                buf[j] = item


def iter_source(name: str, split: str, cfg: dict, drops: Counter0, limit: Optional[int], lic_excl: Counter0):
    s = cfg["sources"][name]
    cache = Path(cfg.get("cache_dir", "~/mimir-decide-data/cache")).expanduser()  # shared across builds
    if name == "tasksource":
        return tasksource.iter_records(split, drops, limit, s.get("revision"), s.get("streaming", True))
    if name == "bekko":
        return bekko.iter_records(split, drops, limit, s.get("revision"), s.get("accept_qualified", False),
                                  excluded=lic_excl)
    if name == "localllama":
        return localllama.iter_records(split, drops, limit, s.get("revision"))
    if name == "massive":
        return massive.iter_records(split, drops, limit, tuple(s["locales"]), s.get("n_options", 8),
                                    cfg["seed"], cache)
    if name == "helpsteer2":
        return helpsteer2.iter_records(split, drops, limit, s.get("revision"))
    raise ValueError(name)


def temperature_quotas(counts: dict[str, int], total: int, alpha: float) -> dict[str, int]:
    """quota_s = min(n_s, c * n_s**alpha) with c chosen by bisection so that sum(quota) ~= total."""
    if sum(counts.values()) <= total:
        return dict(counts)
    lo, hi = 0.0, float(total)
    for _ in range(60):
        c = (lo + hi) / 2
        s = sum(min(n, c * n ** alpha) for n in counts.values())
        lo, hi = (c, hi) if s < total else (lo, c)
    return {k: int(min(n, lo * n ** alpha)) for k, n in counts.items()}


# ----------------------------------------------------------------------------- main build
def build(cfg: dict, limit: Optional[int] = None) -> dict:
    t0 = time.time()
    out = Path(cfg["output_dir"]).expanduser()
    (out / "heldout_tasks").mkdir(parents=True, exist_ok=True)
    (out / "eval" / "test").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    drops: dict[str, Counter0] = defaultdict(Counter0)  # per-source-collection drop reasons
    lic_rows: dict[tuple, int] = defaultdict(int)       # (dataset, source, license, reason) -> rows
    lic_excl_bekko = Counter0()
    held = cfg.get("held_out_sources", [])
    ec = cfg["eval_caps"]
    enabled = [n for n, s in cfg["sources"].items() if s.get("enabled", True)]

    eval_state_hashes: set[str] = set()
    eval_query_hashes: set[str] = set()
    eval_groups: set[str] = set()
    eval_seen: dict[str, int] = defaultdict(int)  # source -> eval rows seen (any licence)
    eval_fp_df: dict[int, int] = defaultdict(int)  # n-gram fingerprint -> number of distinct eval states with it
    eval_fp_src: dict[int, int] = {}                # fingerprint -> eval source id, or -1 if in several sources
    src_ids: dict[str, int] = {}
    ngram_examples: list[dict] = []
    fp_done: set[str] = set()
    boiler_df = cfg.get("ngram_boilerplate_df", 3)  # n-grams in more eval states than this are template text

    def prep(d: Decision, name: str, dr: Counter0) -> Optional[Decision]:
        why = d.validate()
        if why:
            dr.inc(why)
            return None
        if len(d.state) > cfg["max_state_chars"]:
            dr.inc("state_too_long")
            return None
        d2 = cap_options(d, cfg["max_options"])
        if d2 is None:
            dr.inc("too_many_target_options")
        return d2

    # ---- A. eval pass --------------------------------------------------------------
    val_res = Reservoir(ec["validation_per_source"], seed + 1)
    test_res = Reservoir(ec["test_per_source"], seed + 2)
    held_val = Reservoir(ec["heldout_per_source"], seed + 3)
    held_test = Reservoir(ec["heldout_per_source"], seed + 4)
    calib_res = Reservoir(ec["validation_per_source"], seed + 5)
    for name in enabled:
        write_test = cfg["sources"][name].get("write_test", True)
        for split in ("validation", "test"):
            for d in iter_source(name, split, cfg, drops[name], limit, lic_excl_bekko):
                d = prep(d, name, drops[name])
                if d is None:
                    continue
                eval_state_hashes.add(state_hash(d))
                eval_query_hashes.add(query_hash(d))
                eval_groups.add(d.group_id)
                eval_seen[d.source] += 1
                sh = state_hash(d)
                if sh not in fp_done:
                    fp_done.add(sh)
                    sid = src_ids.setdefault(d.source, len(src_ids))
                    for fp in fingerprints(d.state):
                        eval_fp_df[fp] += 1
                        prev = eval_fp_src.get(fp)
                        if prev is None:
                            eval_fp_src[fp] = sid
                        elif prev != sid:
                            eval_fp_src[fp] = -1
                ok, why = licenses.check(d.license) if not d.license.startswith("(excluded") else (False, "excluded")
                lic_rows[(d.dataset, d.source, d.license, f"eval:{why}")] += 1
                if not ok:
                    continue
                if split == "test" and not write_test:
                    continue
                if is_heldout(d.source, held):
                    (held_val if split == "validation" else held_test).add(d.source, d)
                elif split == "test":
                    test_res.add(d.source, d)
                elif bucket(d.group_id, seed + 9) < 0.5:
                    calib_res.add(d.source, d)
                else:
                    val_res.add(d.source, d)

    # ---- B. train pass -------------------------------------------------------------
    train_res = Reservoir(cfg["per_source_cap"], seed + 6)
    seen_full: set[str] = set()
    read_limit = cfg["read_limit_per_source"]
    read_count: dict[str, int] = defaultdict(int)
    for name in enabled:
        dr = drops[name]
        for d in iter_source(name, "train", cfg, dr, limit, lic_excl_bekko):
            if read_count[d.source] >= read_limit:
                dr.inc("read_limit_per_source")
                continue
            read_count[d.source] += 1
            d = prep(d, name, dr)
            if d is None:
                continue
            ok, why = licenses.check(d.license)
            lic_rows[(d.dataset, d.source, d.license, f"train:{why}")] += 1
            if not ok:
                dr.inc(f"license:{why}")
                continue
            if is_heldout(d.source, held):
                dr.inc("held_out_task")
                continue
            if state_hash(d) in eval_state_hashes:
                dr.inc("leak_state_in_eval")
                continue
            if d.group_id in eval_groups:
                dr.inc("leak_group_in_eval")
                continue
            # Content fingerprints are compared against eval rows of OTHER sources only: within one source the
            # upstream split is trusted (templated/synthetic sources legitimately share scenario text across their
            # own train/test); the exact-state and group guards above still apply within a source.
            fps = fingerprints(d.state)
            mine = src_ids.get(d.source, -2)
            hit_fps = [fp for fp in fps if 0 < eval_fp_df.get(fp, 0) <= boiler_df and eval_fp_src[fp] != mine]
            if len(hit_fps) >= 2 or (hit_fps and len(fps) <= 2):
                dr.inc("leak_ngram_in_eval")  # same content as another source's eval state (re-rendered)
                if len(ngram_examples) < 300:
                    other = {v: k for k, v in src_ids.items()}.get(eval_fp_src[hit_fps[0]], "multiple sources")
                    ngram_examples.append({"train_source": d.source, "matched_eval_source": other,
                                           "n_hits": len(hit_fps), "state": d.state[:160]})
                continue
            h = full_hash(d)
            if h in seen_full:
                dr.inc("duplicate")
                continue
            seen_full.add(h)
            train_res.add(d.source, d)

    # ---- C. cap, write, assert ------------------------------------------------------
    counts = {k: len(v) for k, v in train_res.items.items()}
    quotas = temperature_quotas(counts, cfg["max_total_train"], cfg["sampling_alpha"])
    rng = random.Random(seed + 7)
    train: list[Decision] = []
    for k, items in train_res.items.items():
        rng.shuffle(items)
        train += items[: quotas[k]]
    rng.shuffle(train)

    def flat(res: Reservoir) -> list[Decision]:
        return [d for v in res.items.values() for d in v]

    val, calib, test = flat(val_res), flat(calib_res), flat(test_res)
    hv, ht = flat(held_val), flat(held_test)
    write_parquet(train, out / "train.parquet")
    write_parquet(val, out / "validation.parquet")
    write_parquet(calib, out / "calib.parquet")
    write_parquet(hv + ht, out / "heldout_tasks" / "heldout.parquet")
    write_parquet(test, out / "eval" / "test" / "test.parquet")

    # final guard on what was actually written
    eval_written = val + calib + test + hv + ht
    train_states = {state_hash(d) for d in train}
    leaked = train_states & {state_hash(d) for d in eval_written}
    assert not leaked, f"{len(leaked)} train states also appear in eval files"
    assert not train_states & eval_state_hashes, "train states overlap the full eval hash set"
    assert not {d.group_id for d in train} & eval_groups, "train groups overlap eval groups"
    assert not {d.id for d in train} & {d.id for d in eval_written}, "train ids overlap eval ids"
    assert all(d.split == "train" for d in train), "non-train split record in train.parquet"
    assert not any(is_heldout(d.source, held) for d in train), "held-out source present in train"
    assert all(licenses.check(d.license)[0] for d in train), "non-permissive licence in train"

    with open(out / "reports" / "license_exclusions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "source", "license", "split:policy_result", "rows"])
        for k, v in sorted(lic_rows.items()):
            if not k[3].endswith(":ok") and not k[3].endswith(":share_alike"):
                w.writerow([*k, v])
    (out / "reports" / "ngram_leak_examples.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in ngram_examples))
    (out / "reports" / "drops.json").write_text(json.dumps({k: dict(v) for k, v in drops.items()}, indent=1))
    (out / "reports" / "bekko_license_excluded_decisions.json").write_text(json.dumps(dict(lic_excl_bekko), indent=1))

    unmatched = [p for p in held if not any(fnmatch.fnmatch(d.source, p) for d in hv + ht)]
    if unmatched:
        print(f"WARNING: held_out_sources patterns matched no eval rows: {unmatched} "
              f"(expected under --limit smoke builds; investigate otherwise)")
    by_src = lambda recs: {s: sum(1 for d in recs if d.source == s) for s in sorted({d.source for d in recs})}
    manifest = {
        "built_at_unix": int(time.time()),
        "seconds": round(time.time() - t0, 1),
        "limit_per_split": limit,
        "config": cfg,
        "counts": {
            "train": len(train), "validation": len(val), "calib": len(calib),
            "heldout": len(hv) + len(ht), "test": len(test),
        },
        "train_by_kind": {k: sum(1 for d in train if d.kind == k) for k in ("noul", "choice", "score")},
        "train_by_dataset": {k: sum(1 for d in train if d.dataset == k) for k in sorted({d.dataset for d in train})},
        "train_sources": len(counts),
        "train_by_source": by_src(train),
        "heldout_by_source": by_src(hv + ht),
        "eval_hash_counts": {"states": len(eval_state_hashes), "queries": len(eval_query_hashes),
                             "groups": len(eval_groups), "ngram_fingerprints": len(eval_fp_df),
                             "boilerplate_ngrams_ignored": sum(1 for v in eval_fp_df.values() if v > boiler_df)},
        "train_sources_without_eval_rows": sorted(set(counts) - set(eval_seen)),
        "share_alike_train_sources": sorted({d.source for d in train
                                             if licenses.check(d.license)[1] == "share_alike"}),
        "drops": {k: dict(v) for k, v in drops.items()},
        "heldout_patterns_unmatched": unmatched,
        "leakage_assertions": "passed",
    }
    (out / "mixture_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--limit", type=int, default=None, help="max decisions per source per split (smoke builds)")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--sources", nargs="*", default=None, help="restrict to these sources")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config))
    if a.output_dir:
        cfg["output_dir"] = a.output_dir
    if a.sources:
        for n in cfg["sources"]:
            cfg["sources"][n]["enabled"] = n in a.sources
    m = build(cfg, a.limit)
    print(json.dumps({k: m[k] for k in ("counts", "train_by_kind", "train_by_dataset", "train_sources",
                                         "leakage_assertions")}, indent=1))


if __name__ == "__main__":
    main()
