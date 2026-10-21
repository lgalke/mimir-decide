"""Build the Mimir-Decide mixture from the configured sources.

Order matters for leakage control:
  A. EVAL PASS   read validation + test of every source; hash ALL of it (every licence) into the leakage guard
                 and keep a capped, licence-allowed sample for the eval files.
  B. TRAIN PASS  read train splits only; drop rows that are invalid, too long, licence-excluded, from held-out
                 tasks, or that leak (state hash or group id seen in ANY eval split of ANY source); dedup; cap.
  C. WRITE       global temperature-based cap, parquet shards, manifest, reports, final zero-overlap assertion.

Outputs (under output_dir): train.parquet, validation.parquet, calib.parquet, heldout_tasks/*.parquet,
eval/test/*.parquet, mixture_manifest.json, reports/{license_exclusions.csv,drops.json}.

Mixture v1 (D31) adds, all optional and off in configs/data.yaml so that v0 stays reproducible: separate length limits
for the eval and train pass, a tasksource config (`config`, `eval_config`, `guard_eval_configs`), exclusion lists,
annotation-tolerant licence parsing, a `sampling` section (family caps, kind floors, binary cap, nested named sizes)
and `frozen_eval_from` (the written eval files must equal those of an earlier mixture).
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import yaml

from . import licenses, paths, sampling
from .convert import bekko, helpsteer2, localllama, massive, tasksource
from .convert.common import Counter0
from .schema import Decision, bucket, fingerprints, full_hash, query_hash, read_parquet, state_hash, write_parquet


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


def load_exclusions(cfg: dict) -> list[str]:
    """Source patterns that must never enter train: `exclude_sources` plus the `held_out_sources` /
    `task_novel_sources` / `exclude_sources` keys of the YAML files in `exclude_sources_from` (for example the
    pre-registered extra evaluation sources of E12)."""
    pats = list(cfg.get("exclude_sources", []))
    for f in cfg.get("exclude_sources_from", []):
        data = yaml.safe_load(Path(paths.resolve(f)).read_text()) or {}
        for key in ("held_out_sources", "task_novel_sources", "exclude_sources"):
            pats += list(data.get(key, []))
    return sorted(set(pats))


def is_excluded(source: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(source, p) for p in patterns)


EVAL_FILES = ("validation.parquet", "calib.parquet", "heldout_tasks/heldout.parquet", "eval/test/test.parquet")


def eval_digest(path: Path) -> str:
    """Order-independent digest of an eval file: id, state hash, options and target of every decision."""
    if not Path(path).exists():
        return "missing"
    rows = sorted((d.id, state_hash(d), tuple(d.options), tuple(round(t, 9) for t in d.target))
                  for d in read_parquet(path))
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def check_frozen_eval(out: Path, ref: Path) -> dict:
    """The written eval files must equal those of the reference mixture (D31: freeze the evaluation)."""
    res = {f: (eval_digest(out / f) == eval_digest(ref / f)) for f in EVAL_FILES}
    bad = [f for f, ok in res.items() if not ok]
    assert not bad, f"eval files differ from the frozen reference {ref}: {bad}"
    return res


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


def iter_source(name: str, split: str, cfg: dict, drops: Counter0, limit: Optional[int], lic_excl: Counter0,
                config: Optional[str] = None):
    s = cfg["sources"][name]
    cache = Path(cfg.get("cache_dir") or paths.data_root() / "cache").expanduser()  # shared across builds
    if name == "tasksource":
        # `config` (train) and `eval_config` (validation/test) can differ: the evaluation rows of v1 are read from the
        # `default` config so that they stay identical to v0 while the training rows come from `full`.
        cfg_name = config or (s.get("config", "default") if split == "train"
                              else s.get("eval_config", s.get("config", "default")))
        return tasksource.iter_records(split, drops, limit, s.get("revision"), s.get("streaming", True), cfg_name)
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


def sample_v1(items_by_source: dict, cfg: dict, seed: int):
    """Mixture v1 sampling (D31): temperature quotas with a per-source and per-family cap, a fixed priority order inside
    every source, boosts for the kind floors and the binary cap, and nested named sizes.

    The full set has `total` decisions (min of the pool and `sampling.total`). Named sizes are drawn from the full set:
    their quotas are solved on the per-source counts of the full set and use the same priority order and boosts, so
    every subset is contained in the full set (and in every larger subset)."""
    sc = cfg["sampling"]
    alpha = sc.get("alpha", 0.5)
    fam_patterns = sc.get("families", {})
    names = sorted(items_by_source)
    fam = {k: sampling.family_of(k, items_by_source[k][0].dataset, fam_patterns) for k in names if items_by_source[k]}
    names = [k for k in names if items_by_source[k]]
    counts = {k: len(items_by_source[k]) for k in names}
    arrays = {}
    for k in names:
        its = items_by_source[k]
        arrays[k] = sampling.SourceArrays(
            sampling.priorities([d.id for d in its], seed + 8),
            np.array([sampling.KIND_CODE[d.kind] for d in its], dtype=np.int8),
            np.array([len(d.options) == 2 for d in its], dtype=bool))
    total = min(sum(counts.values()), int(sc.get("total", sum(counts.values()))))
    share = sc.get("max_source_share")

    def quotas_for(avail: dict, n: int) -> dict:
        return sampling.solve_quotas(avail, n, alpha, int(share * n) if share else None, fam, sc.get("family_caps"))

    q_full = quotas_for(counts, total)
    boosts, bin_pen, comp, ok = sampling.solve_boosts(arrays, q_full, sc.get("kind_floors"), sc.get("max_binary_share"))
    if not ok:
        print(f"WARNING: composition targets not met for the full set: {comp}")

    def draw(quota: dict) -> list:
        out = []
        for k in names:
            idx = sampling.select(arrays[k], quota.get(k, 0), boosts, bin_pen)
            out += [items_by_source[k][i] for i in idx]
        return out

    full = draw(q_full)
    report = {"total_requested": sc.get("total"), "full": len(full), "alpha": alpha, "max_source_share": share,
              "family_caps": sc.get("family_caps"), "boosts": boosts, "binary_penalty": bin_pen,
              "kind_floors": sc.get("kind_floors"), "max_binary_share": sc.get("max_binary_share"),
              "composition": {"full": comp}, "converged": ok, "sizes": {}}
    subsets = {}
    sources_full = {k: arrays[k] for k in names}
    for name, n in (sc.get("sizes") or {}).items():
        if n is None or n >= len(full):
            continue
        q = quotas_for(q_full, int(n))
        recs = draw(q)
        subsets[name] = recs
        report["sizes"][name] = {"requested": int(n), "decisions": len(recs), "sources": sum(1 for v in q.values() if v)}
        report["composition"][name] = sampling.composition(sources_full, q, boosts, bin_pen)
        for kind, floor in (sc.get("kind_floors") or {}).items():
            if report["composition"][name][kind] < floor - 0.03:
                print(f"WARNING: subset {name}: {kind} share {report['composition'][name][kind]:.3f} below floor {floor}")
    return full, subsets, report


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
    eval_max_chars = cfg.get("eval_max_state_chars", cfg["max_state_chars"])    # v0: one limit for both passes
    train_max_chars = cfg.get("train_max_state_chars", cfg["max_state_chars"])
    strip_ann = bool(cfg.get("license_strip_annotations", False))                # D32; the eval pass never uses it
    guard_long_eval = bool(cfg.get("guard_long_eval_states", False))
    excluded = load_exclusions(cfg)
    newly_admitted: dict[tuple, int] = defaultdict(int)
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

    def prep(d: Decision, name: str, dr: Counter0, max_chars: int) -> Optional[Decision]:
        why = d.validate()
        if why:
            dr.inc(why)
            return None
        if len(d.state) > max_chars:
            dr.inc("state_too_long")
            return None
        d2 = cap_options(d, cfg["max_options"])
        if d2 is None:
            dr.inc("too_many_target_options")
        return d2

    def guard_add(d: Decision) -> None:
        """Add one eval decision to the leakage guards (exact state, query, group, content fingerprints)."""
        eval_state_hashes.add(state_hash(d))
        eval_query_hashes.add(query_hash(d))
        eval_groups.add(d.group_id)
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
                d0 = d
                d = prep(d, name, drops[name], eval_max_chars)
                if d is None:
                    # v1: an eval state that is only too long for the written eval files still guards the train set
                    if guard_long_eval and d0.validate() is None:
                        guard_add(d0)
                    continue
                guard_add(d)
                eval_seen[d.source] += 1
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

    for name in enabled:  # eval rows of further configs (for example tasksource `full`): guards only, never written
        for extra in cfg["sources"][name].get("guard_eval_configs", []):
            for split in ("validation", "test"):
                for d in iter_source(name, split, cfg, drops[f"{name}:guard:{extra}"], limit, lic_excl_bekko, extra):
                    if d.validate() is None:
                        guard_add(d)

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
            d = prep(d, name, dr, train_max_chars)
            if d is None:
                continue
            if is_excluded(d.source, excluded):
                dr.inc("excluded_source")
                continue
            ok, why = licenses.check(d.license, None, strip_ann)
            lic_rows[(d.dataset, d.source, d.license, f"train:{why}")] += 1
            if not ok:
                dr.inc(f"license:{why}")
                continue
            if strip_ann and not licenses.check(d.license)[0]:
                newly_admitted[(d.dataset, d.source, d.license)] += 1
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
    rng = random.Random(seed + 7)
    train: list[Decision] = []
    subsets: dict[str, list[Decision]] = {}
    sampling_report: dict = {}
    if cfg.get("sampling"):
        train, subsets, sampling_report = sample_v1(train_res.items, cfg, seed)
        counts = dict(Counter(d.source for d in train))
    else:  # v0 behaviour, unchanged
        quotas = temperature_quotas(counts, cfg["max_total_train"], cfg["sampling_alpha"])
        for k, items in train_res.items.items():
            rng.shuffle(items)
            train += items[: quotas[k]]
    rng.shuffle(train)

    def flat(res: Reservoir) -> list[Decision]:
        return [d for v in res.items.values() for d in v]

    val, calib, test = flat(val_res), flat(calib_res), flat(test_res)
    hv, ht = flat(held_val), flat(held_test)
    write_parquet(train, out / "train.parquet")
    subset_files = {}
    for name, recs in subsets.items():
        fn = f"train_{name}.parquet"
        write_parquet(recs, out / fn)
        subset_files[name] = {"file": fn, "decisions": len(recs)}
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
    assert all(licenses.check(d.license, None, strip_ann)[0] for d in train), "non-permissive licence in train"
    full_ids = {d.id for d in train}
    for name, recs in subsets.items():
        assert {d.id for d in recs} <= full_ids, f"subset {name} is not nested in the full train set"
        assert not {state_hash(d) for d in recs} & eval_state_hashes, f"subset {name} overlaps the eval hash set"
    assert not any(is_excluded(d.source, excluded) for d in train), "excluded source present in train"
    frozen = None
    if cfg.get("frozen_eval_from"):
        frozen = check_frozen_eval(out, Path(cfg["frozen_eval_from"]).expanduser())

    with open(out / "reports" / "license_exclusions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "source", "license", "split:policy_result", "rows"])
        for k, v in sorted(lic_rows.items()):
            if not k[3].endswith(":ok") and not k[3].endswith(":share_alike"):
                w.writerow([*k, v])
    (out / "reports" / "ngram_leak_examples.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in ngram_examples))
    (out / "reports" / "drops.json").write_text(json.dumps({k: dict(v) for k, v in drops.items()}, indent=1))
    with open(out / "reports" / "license_newly_admitted.csv", "w", newline="") as f:  # D32: for the owner's review
        w = csv.writer(f)
        w.writerow(["dataset", "source", "license", "train_rows_passing_licence_stage"])
        for k, v in sorted(newly_admitted.items()):
            w.writerow([*k, v])
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
        "excluded_source_patterns": excluded,
        "license_strip_annotations": strip_ann,
        "newly_admitted_by_annotation_stripping": sum(newly_admitted.values()),
        "subsets": subset_files,
        "sampling": sampling_report,
        "frozen_eval_identical_to_reference": frozen,
    }
    (out / "mixture_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--limit", type=int, default=None, help="max decisions per source per split (smoke builds)")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--sources", nargs="*", default=None, help="restrict to these sources")
    ap.add_argument("--set", nargs="*", default=[], help="top-level overrides key=value (YAML-parsed), for trial builds")
    a = ap.parse_args(argv)
    cfg = paths.load_config(a.config)
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = paths.resolve(yaml.safe_load(v))
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
