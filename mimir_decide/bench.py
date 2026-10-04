"""Latency / throughput of a trained decision model on validation decisions (one decision = one sequence).

    python -m mimir_decide.bench --run_dir RUN --data_dir MIXTURE [--batch_sizes 1 8 32] [--n 256]

Note: every question is its own forward pass here. Jev evaluates several questions about one state in parallel for
roughly the cost of one; this model does not (yet), so cost grows with the number of questions per state.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from .inference import device_auto, predict_logits
from .model import load_model
from .schema import read_parquet


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--checkpoint", default="best")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args(argv)
    device = device_auto()
    run = Path(a.run_dir).expanduser()
    model, tok, meta = load_model(run / a.checkpoint, device)
    decs = read_parquet(Path(a.data_dir).expanduser() / "validation.parquet")[: a.n]
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    out = {"device": device, "n": len(decs), "checkpoint": a.checkpoint, "results": {}}
    for bs in a.batch_sizes:
        predict_logits(model, tok, meta, decs[: max(bs, 8)], bs, device, autocast=device == "cuda")  # warm-up
        times = []
        for _ in range(a.repeats):
            sync(); t = time.perf_counter()
            predict_logits(model, tok, meta, decs, bs, device, autocast=device == "cuda")
            sync(); times.append(time.perf_counter() - t)
        t = statistics.median(times)
        out["results"][str(bs)] = {"decisions_per_s": round(len(decs) / t, 2), "ms_per_decision": round(1000 * t / len(decs), 2)}
        if device == "cuda":
            out["results"][str(bs)]["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    (run / "eval").mkdir(exist_ok=True)
    (run / "eval" / "latency.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
