"""Evaluate a run on validation / heldout / test.

`--split test` is a deliberate, recorded act: it requires --final and appends to <run_dir>/eval/test_access_log.jsonl.
Sources that Mimir v1.5 has (by name) seen in training are flagged if reports/mimir_overlap.json exists.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .inference import device_auto, predict_logits
from .metrics import evaluate
from .model import load_model
from .schema import read_parquet

PATHS = {"validation": "validation.parquet", "heldout": "heldout_tasks/heldout.parquet",
         "test": "eval/test/test.parquet"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--checkpoint", default="best")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", choices=list(PATHS), default="validation")
    ap.add_argument("--final", action="store_true", help="required for --split test")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--no_calibration", action="store_true")
    ap.add_argument("--order_seed", type=int, default=None,
                    help="present options in a seeded random order (robustness to option position)")
    a = ap.parse_args(argv)
    if a.split == "test" and not a.final:
        sys.exit("refusing to evaluate on the test split without --final (test access is logged and meant to be rare)")
    run, data = Path(a.run_dir).expanduser(), Path(a.data_dir).expanduser()
    device = device_auto()
    model, tok, meta = load_model(run / a.checkpoint, device)
    recs = read_parquet(data / PATHS[a.split])
    if a.split != "test":
        assert all(d.split in ("validation", "test") for d in recs)
    lg, kept = predict_logits(model, tok, meta, recs, a.batch_size, device, autocast=device == "cuda",
                              perm_seed=a.order_seed)
    cal = run / "calibration.json"
    temps = None if a.no_calibration or not cal.exists() else json.loads(cal.read_text())["temperatures"]
    res = evaluate(lg, kept, temps)
    overlap = data / "reports" / "mimir_overlap.json"
    out = {"split": a.split, "order_seed": a.order_seed, "checkpoint": a.checkpoint, "calibrated": temps is not None, "temperatures": temps,
           "n_records": len(recs), "n_evaluated": len(kept), "n_skipped_too_long": len(recs) - len(kept),
           "results": res}
    if overlap.exists():
        seen = set(json.loads(overlap.read_text()).get("sources_seen_by_mimir", []))
        out["sources_seen_by_mimir"] = sorted(seen & {k.split(":", 1)[1] for k in res if k.startswith("source:")})
    (run / "eval").mkdir(exist_ok=True)
    (run / "eval" / (f"{a.split}.json" if a.order_seed is None else f"{a.split}_order{a.order_seed}.json")).write_text(json.dumps(out, indent=1))
    if a.split == "test":
        with open(run / "eval" / "test_access_log.jsonl", "a") as f:
            f.write(json.dumps({"time": int(time.time()), "checkpoint": a.checkpoint,
                                "meta": meta, "all": res["all"]}) + "\n")
    print(json.dumps({"all": res["all"], **{k: v for k, v in res.items() if k.startswith("kind:")}}, indent=1))


if __name__ == "__main__":
    main()
