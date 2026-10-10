"""Compare evaluation JSONs of several runs (e.g. slot vs letter baseline), split by 'seen by Mimir' or not.

    python -m mimir_decide.compare RUN_DIR [RUN_DIR ...] [--split validation] [--file validation.json]

`sources_seen_by_mimir` comes from the name-level audit (see docs/design/dataset-selection.md): numbers on seen
sources measure the format conversion and readout, not generalisation. accuracy / NLL / Brier are weighted by n
when pooled; ECE is not additive, so it is reported only for the groups evaluate.py computed (all, kind:*).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def pool(results: dict, sources: list[str]) -> dict:
    """n-weighted acc/nll/brier over the given `source:*` entries."""
    n = acc = nll = brier = 0.0
    for s in sources:
        r = results.get(f"source:{s}")
        if r:
            n += r["n"]; acc += r["n"] * r["acc"]; nll += r["n"] * r["nll"]; brier += r["n"] * r["brier"]
    return {"n": int(n), "acc": acc / n, "nll": nll / n, "brier": brier / n} if n else {"n": 0}


def split_by_seen(ev: dict) -> dict:
    res = ev["results"]
    srcs = [k.split(":", 1)[1] for k in res if k.startswith("source:")]
    seen = set(ev.get("sources_seen_by_mimir", []))
    return {"seen_by_mimir": pool(res, [s for s in srcs if s in seen]),
            "not_flagged": pool(res, [s for s in srcs if s not in seen])}


def fmt(r: dict) -> str:
    if not r or not r.get("n"):
        return "n=0"
    ece = f" ece={r['ece']:.3f}" if "ece" in r else ""
    return f"n={r['n']:<6} acc={r['acc']:.3f} nll={r['nll']:.3f} brier={r['brier']:.3f}{ece}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--file", default="validation.json", help="eval JSON name inside <run>/eval/")
    a = ap.parse_args(argv)
    evs = {}
    for r in a.run_dirs:
        f = Path(r).expanduser() / "eval" / a.file
        if not f.exists():
            have = sorted(x.name for x in f.parent.glob("*.json")) if f.parent.exists() else []
            raise SystemExit(
                f"compare: {f} does not exist.\n"
                f"  Files in {f.parent}: {', '.join(have) if have else '(none; run evaluate for this run first)'}\n"
                f"  Create it with: python -m mimir_decide.evaluate --run_dir {r} --data_dir <mixture> --split validation"
                + (" --checkpoint zero-shot" if "zero" in str(r) else ""))
        evs[Path(r).expanduser()] = json.loads(f.read_text())
    groups = ["all", "kind:noul", "kind:choice", "kind:score"]
    for g in groups:
        print(f"\n[{g}]")
        for run, ev in evs.items():
            print(f"  {run.name:<24} {fmt(ev['results'].get(g, {}))}")
    for part in ("seen_by_mimir", "not_flagged"):
        print(f"\n[{part}] (pooled over sources; ECE omitted)")
        for run, ev in evs.items():
            print(f"  {run.name:<24} {fmt(split_by_seen(ev)[part])}")
    print("\ncalibrated:", {r.name: ev["calibrated"] for r, ev in evs.items()},
          "| skipped_too_long:", {r.name: ev["n_skipped_too_long"] for r, ev in evs.items()})


if __name__ == "__main__":
    main()
