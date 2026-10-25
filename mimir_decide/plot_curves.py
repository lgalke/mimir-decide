"""Plot validation curves of training runs (needs matplotlib, an optional dependency: pip install matplotlib).

    python -m mimir_decide.plot_curves runs/baseline-v0 runs/scalar-v0 --labels "letter baseline" "scalar head" --out curves.png

A run is a run directory (reads train_log.jsonl) or a text log of `python -m mimir_decide.train` (lines with
`'val_nll': ...`). The validation sample is the first 2,000 decisions of validation.parquet, so one standard error
of the accuracy is about 0.9 points: differences smaller than about 1.5 points between two curves are noise.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


def read_evals(path, key: str = "eval") -> list[dict]:
    """Evaluation records {step, val_nll, val_acc, val_ece} of a run directory, a train_log.jsonl or a text log.
    `key` selects `eval` (training weights) or `eval_ema` (averaged weights, runs with ema_decay)."""
    p = Path(path)
    if p.is_dir():
        p = p / "train_log.jsonl"
    out = []
    for line in p.read_text().splitlines():
        if "val_nll" not in line:
            continue
        m = re.search(r"\{.*\}", line)
        if not m:
            continue
        try:
            rec = json.loads(m.group(0))
        except ValueError:
            try:
                rec = ast.literal_eval(m.group(0))
            except (ValueError, SyntaxError):
                continue
        rec = rec.get(key, rec if key == "eval" else {})
        if "val_nll" in rec:
            out.append(rec)
    return sorted({r["step"]: r for r in out}.values(), key=lambda r: r["step"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories, train_log.jsonl files or text logs")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ema", action="store_true", help="also plot the averaged-weights curve (dashed) of runs that have one")
    ap.add_argument("--title", default="Validation curves (first 2,000 validation decisions)")
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = a.labels or [Path(r).name for r in a.runs]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for run, label in zip(a.runs, labels):
        ev = read_evals(run)
        x = [r["step"] for r in ev]
        for axis, key in zip(ax, ("val_nll", "val_acc", "val_ece")):
            line, = axis.plot(x, [r[key] for r in ev], marker="o", ms=4, label=f"{label} (n={len(ev)})")
            if a.ema:
                ev_e = read_evals(run, "eval_ema")
                if ev_e:
                    axis.plot([r["step"] for r in ev_e], [r[key] for r in ev_e], marker="s", ms=3, ls="--",
                              color=line.get_color(), label=f"{label}, averaged weights")
    for axis, title in zip(ax, ("Validation NLL (lower is better)", "Validation accuracy", "Validation ECE (lower is better)")):
        axis.set_title(title)
        axis.set_xlabel("optimizer step")
        axis.grid(alpha=0.3)
    ax[1].text(0.02, 0.03, "1 standard error ~ 0.9 points (n = 2,000)", transform=ax[1].transAxes, fontsize=8, alpha=0.7)
    ax[0].legend()
    fig.suptitle(a.title)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
