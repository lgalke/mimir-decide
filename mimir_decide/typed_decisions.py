"""Score a model on the LocalLLaMA/typed-decisions benchmark (full test split: 400 cases, 2,000 decisions).

    python -m mimir_decide.typed_decisions --run_dir runs/slot-v0 --final            # checkpoint 'best'; writes calibrated AND raw from one pass
    python -m mimir_decide.typed_decisions --run_dir runs/letter-zeroshot --checkpoint zero-shot --final
    python -m mimir_decide.typed_decisions --references                               # Uniform and Prior only
    python -m mimir_decide.typed_decisions --score_predictions anna.json --name anna --final   # another approach, same scorer

Metrics (per decision, then averaged over all 2,000 decisions; the benchmark README does not define them, so the
definitions are fixed here and written into every output):
  accuracy  the predicted option with the highest probability (ties: first) equals the gold option with the highest probability
  kl        KL(gold || pred) in natural log, pred clipped at 1e-12     (reproduces the published Uniform row: 0.444)
  brier     sum over options of (pred - gold)^2                         (reproduces the published Uniform row: 0.238)
  ece       15 equal-width bins on the top-1 confidence against accuracy
  nll       cross-entropy of the prediction against the gold distribution (extra)
Accuracy and ECE of the published Uniform/Prior rows could NOT be reproduced exactly (Uniform: 0.278 here against 0.308
published); the benchmark README says ECE is computed differently by different submitters. Compare approaches only
with the SAME scorer: the per-decision probabilities are saved so that any other approach can be rescored.

Every decision is answered on its own (one forward pass per question), not all five questions of a case in one request.
The test split is accessed through `--final` and logged like every test access (D11).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .convert import localllama
from .convert.common import Counter0
from .schema import Decision

REPO_N_CASES, REPO_N_DECISIONS = 400, 2000


def load_decisions(split: str) -> list[Decision]:
    """All decisions of the benchmark split in the training format (state, question, options, gold distribution)."""
    from datasets import load_dataset

    ds = load_dataset(localllama.REPO, "all", split=split)
    drops = Counter0()
    out = []
    for r in ds:
        out += localllama.row_to_decisions(r, "test" if split == "test" else "train", drops)
    return out


def option_keys(d: Decision) -> list[str]:
    """Benchmark keys of the options: noul true/false, choice criterion ids, score levels 0..n-1."""
    if d.kind == "noul":
        return ["true", "false"]
    if d.kind == "score":
        return [str(i) for i in range(len(d.options))]
    return [o.split(": ", 1)[0] for o in d.options]


def workflow(d: Decision) -> str:
    return d.source.split("/", 1)[1]


def question(d: Decision) -> str:
    return d.id.rsplit(":", 1)[1]


# ------------------------------------------------------------------------------------------------ metrics
def decision_metrics(p: np.ndarray, g: np.ndarray) -> dict:
    p = np.asarray(p, float)
    g = np.asarray(g, float)
    pc = np.clip(p, 1e-12, 1.0)
    pos = g > 0
    return {
        "acc": float(p.argmax() == g.argmax()),
        "kl": float(np.sum(g[pos] * np.log(g[pos] / pc[pos]))),
        "brier": float(np.sum((p - g) ** 2)),
        "nll": float(-np.sum(g[pos] * np.log(pc[pos]))),
        "conf": float(p.max()),
    }


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(out)


def aggregate(rows: list[dict]) -> dict:
    conf = np.array([r["conf"] for r in rows])
    acc = np.array([r["acc"] for r in rows])
    return {"n": len(rows), "accuracy": float(acc.mean()), "kl_from_gold": float(np.mean([r["kl"] for r in rows])),
            "brier": float(np.mean([r["brier"] for r in rows])), "ece": ece(conf, acc),
            "nll": float(np.mean([r["nll"] for r in rows])), "mean_conf": float(conf.mean())}


def score(probs: list[np.ndarray], decs: list[Decision]) -> dict:
    rows = [decision_metrics(p, np.array(d.target)) for p, d in zip(probs, decs)]
    groups: dict[str, dict[str, list]] = {"by_kind": defaultdict(list), "by_workflow": defaultdict(list),
                                          "by_question": defaultdict(list)}
    for r, d in zip(rows, decs):
        groups["by_kind"][d.kind].append(r)
        groups["by_workflow"][workflow(d)].append(r)
        groups["by_question"][f"{workflow(d)}/{question(d)}"].append(r)
    return {"overall": aggregate(rows), **{k: {n: aggregate(v) for n, v in sorted(g.items())} for k, g in groups.items()}}


def references(train: list[Decision], test: list[Decision]) -> dict:
    """Uniform and Prior (mean gold distribution of the train split per workflow and question)."""
    uni = [np.ones(len(d.target)) / len(d.target) for d in test]
    acc: dict = defaultdict(list)
    for d in train:
        acc[(workflow(d), question(d))].append(np.array(d.target))
    prior = {k: np.mean(v, 0) for k, v in acc.items()}
    return {"uniform": score(uni, test), "prior_mean_gold": score([prior[(workflow(d), question(d))] for d in test], test),
            "published_uniform": {"accuracy": 0.308, "kl_from_gold": 0.444, "brier": 0.238, "ece": 0.169},
            "published_prior": {"accuracy": 0.470, "kl_from_gold": 0.347, "brier": 0.189, "ece": 0.088},
            "note": "KL and Brier of Uniform match the published row exactly; accuracy, ECE and the Prior rows do not "
                    "(the README does not define them), so only compare scores made with this scorer."}


# ------------------------------------------------------------------------------------------------ external predictions
_BOOL = {"true": "true", "yes": "true", "1": "true", "false": "false", "no": "false", "0": "false"}


def parse_external(raw, decs: list[Decision]) -> list[np.ndarray]:
    """Align predictions of another approach with the benchmark's decisions, independent of its option order.

    Accepted: a list of entries, or a dict id -> entry. An entry has an id `<case id>:<question>` (the row `id` of the
    dataset and the question name) and either `keys` + `probs` (this module's own output format) or
    `probabilities` as {option key: probability}. Option keys are the criterion ids for choice questions, the level
    "0".."n-1" for scores, and true/false (also yes/no) for noul. Each distribution is renormalised to sum to 1."""
    if isinstance(raw, dict):
        raw = [{"id": k, **(v if isinstance(v, dict) else {"probabilities": v})} for k, v in raw.items()]
    by_id = {}
    for e in raw:
        if "probabilities" in e:
            m = e["probabilities"]
            if isinstance(m, dict):
                pairs = list(m.items())
            else:
                raise ValueError(f"{e['id']}: 'probabilities' must be a dict option key -> probability")
        else:
            pairs = list(zip(e["keys"], e["probs"]))
        by_id[e["id"]] = {str(k).strip().lower() if str(k).strip().lower() in _BOOL else str(k): float(v) for k, v in pairs}
    out, missing = [], []
    for d in decs:
        m = by_id.get(d.id)
        if m is None:
            missing.append(d.id)
            continue
        keys = option_keys(d)
        vals = []
        for k in keys:
            alt = [k] + ([a for a, t in _BOOL.items() if t == k] if d.kind == "noul" else [])
            hit = next((m[a] for a in alt if a in m), None)
            if hit is None:
                raise ValueError(f"{d.id}: no probability for option key {k!r} (found {sorted(m)})")
            vals.append(hit)
        p = np.array(vals, float)
        if not np.all(p >= 0) or p.sum() <= 0:
            raise ValueError(f"{d.id}: probabilities must be non-negative and sum to more than 0")
        out.append(p / p.sum())
    if missing:
        raise ValueError(f"{len(missing)} of {len(decs)} decisions have no prediction, first: {missing[:3]}")
    return out


# ------------------------------------------------------------------------------------------------ model scoring
def softmax_T(logits: np.ndarray, T: float) -> np.ndarray:
    z = logits / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def predict(run_dir: Path, checkpoint: str, decs: list[Decision], batch_size: int):
    from .inference import device_auto, predict_logits
    from .model import load_model

    device = device_auto()
    model, tok, meta = load_model(run_dir / checkpoint, device)
    logits, kept = predict_logits(model, tok, meta, decs, batch_size, device, autocast=device == "cuda")
    assert len(kept) == len(decs), f"{len(decs) - len(kept)} decisions were skipped (too long); the benchmark needs all"
    return logits


def write_variant(run: Path, stem: str, logits, decs, temps, checkpoint: str, limit):
    probs = [softmax_T(l, (temps or {}).get(d.kind, 1.0)) for l, d in zip(logits, decs)]
    res = score(probs, decs)
    out = {"benchmark": "LocalLLaMA/typed-decisions", "split": "test", "checkpoint": checkpoint,
           "calibrated": temps is not None, "temperatures": temps, "limit": limit,
           "scorer": "mimir_decide.typed_decisions (definitions in the module docstring)", **res}
    (run / "eval" / f"{stem}.json").write_text(json.dumps(out, indent=1))
    preds = [{"id": d.id, "kind": d.kind, "keys": option_keys(d), "probs": [round(float(x), 6) for x in p]}
             for p, d in zip(probs, decs)]
    (run / "eval" / f"{stem}_predictions.json").write_text(json.dumps(preds, separators=(",", ":")))
    o = res["overall"]
    print(f"typed-decisions test, {o['n']} decisions, calibrated={temps is not None}: accuracy {o['accuracy']:.3f}  "
          f"KL {o['kl_from_gold']:.3f}  Brier {o['brier']:.3f}  ECE {o['ece']:.3f}  NLL {o['nll']:.3f}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir")
    ap.add_argument("--checkpoint", default="best")
    ap.add_argument("--no_calibration", action="store_true", help="write only the raw version (default: calibrated and raw, from one pass)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--references", action="store_true", help="only compute Uniform and Prior")
    ap.add_argument("--score_predictions", metavar="FILE", help="score another approach's predictions (JSON) with this scorer")
    ap.add_argument("--name", default="external", help="name for --score_predictions; results go to runs/external-<name>/eval/")
    ap.add_argument("--final", action="store_true", help="required: this reads the benchmark's test split")
    ap.add_argument("--limit", type=int, default=None, help="first N decisions (smoke test only)")
    a = ap.parse_args(argv)
    if not a.final:
        raise SystemExit("refusing to read the benchmark's test split without --final (test access is logged, D11)")
    test = load_decisions("test")
    if a.limit is None:
        assert len(test) == REPO_N_DECISIONS, f"expected {REPO_N_DECISIONS} decisions, got {len(test)}"
    if a.score_predictions:
        probs = parse_external(json.load(open(a.score_predictions)), test)
        run = Path("runs") / f"external-{a.name}"
        (run / "eval").mkdir(parents=True, exist_ok=True)
        write_variant(run, "typed_decisions_test", [np.log(np.clip(p, 1e-12, 1)) for p in probs], test, None, a.score_predictions, None)
        (run / "eval" / "typed_decisions_test.json").write_text(
            (run / "eval" / "typed_decisions_test.json").read_text().replace('"calibrated": false', '"calibrated": "n/a (predictions as given)"'))
        res = json.loads((run / "eval" / "typed_decisions_test.json").read_text())
        for k, v in res["by_kind"].items():
            print(f"  {k:<7} n={v['n']:<5} accuracy {v['accuracy']:.3f}  KL {v['kl_from_gold']:.3f}  Brier {v['brier']:.3f}")
        for k, v in res["by_workflow"].items():
            print(f"  {k:<28} n={v['n']:<5} accuracy {v['accuracy']:.3f}  KL {v['kl_from_gold']:.3f}  Brier {v['brier']:.3f}")
        return
    if a.references or not a.run_dir:
        print(json.dumps(references(load_decisions("train"), test), indent=1))
        return
    decs = test[: a.limit] if a.limit else test
    run = Path(a.run_dir).expanduser()
    (run / "eval").mkdir(exist_ok=True)
    logits = predict(run, a.checkpoint, decs, a.batch_size)          # ONE pass over the test split, then both versions
    cal = run / "calibration.json"
    temps = json.loads(cal.read_text())["temperatures"] if cal.exists() else None
    res = None
    if temps is not None and not a.no_calibration:
        res = write_variant(run, "typed_decisions_test", logits, decs, temps, a.checkpoint, a.limit)
    raw = write_variant(run, "typed_decisions_test_uncal", logits, decs, None, a.checkpoint, a.limit)
    res = res or raw
    with open(run / "eval" / "test_access_log.jsonl", "a") as f:
        f.write(json.dumps({"time": int(time.time()), "checkpoint": a.checkpoint, "what": "typed-decisions benchmark test",
                            "overall": res["overall"]}) + "\n")
    for k, v in res["by_kind"].items():
        print(f"  {k:<7} n={v['n']:<5} accuracy {v['accuracy']:.3f}  KL {v['kl_from_gold']:.3f}  Brier {v['brier']:.3f}")
    for k, v in res["by_workflow"].items():
        print(f"  {k:<28} n={v['n']:<5} accuracy {v['accuracy']:.3f}  KL {v['kl_from_gold']:.3f}  Brier {v['brier']:.3f}")


if __name__ == "__main__":
    main()
