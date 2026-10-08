"""Fit one temperature per decision kind on calib.parquet (a split of VALIDATION data, never train/test)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .inference import device_auto, predict_logits
from .metrics import evaluate, fit_temperature
from .model import load_model
from .schema import read_parquet


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--checkpoint", default="best")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="use a seeded random sample of N records (smoke tests only)")
    a = ap.parse_args(argv)
    device = device_auto()
    model, tok, meta = load_model(Path(a.run_dir).expanduser() / a.checkpoint, device)
    calib = read_parquet(Path(a.data_dir).expanduser() / "calib.parquet")
    assert all(d.split == "validation" for d in calib), "calibration data must come from validation"
    if a.limit is not None and a.limit < len(calib):
        calib = random.Random(0).sample(calib, a.limit)
    lg, kept = predict_logits(model, tok, meta, calib, a.batch_size, device, autocast=device == "cuda")
    temps, report = {}, {}
    for kind in ("noul", "choice", "score"):
        idx = [i for i, d in enumerate(kept) if d.kind == kind]
        temps[kind] = fit_temperature([lg[i] for i in idx], [np.array(kept[i].target) for i in idx])
        report[kind] = {"n": len(idx), "T": temps[kind]}
    before, after = evaluate(lg, kept)["all"], evaluate(lg, kept, temps)["all"]
    out = {"temperatures": temps, "per_kind": report, "checkpoint": a.checkpoint,
           "calib_ece_before": before["ece"], "calib_ece_after": after["ece"],
           "calib_nll_before": before["nll"], "calib_nll_after": after["nll"]}
    Path(a.run_dir).expanduser().joinpath("calibration.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
