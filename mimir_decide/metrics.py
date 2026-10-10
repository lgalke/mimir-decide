"""Decision metrics: accuracy, NLL, Brier, ECE, ordinal MAE/Spearman, selective risk, temperature fitting."""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np

from .schema import Decision


def softmax(x: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = x / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def top1_correct(p: np.ndarray, t: np.ndarray) -> bool:
    """Prediction counts as correct if it picks (one of) the option(s) with maximal target mass."""
    return bool(t[int(p.argmax())] >= t.max() - 1e-9)


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(out)


def reliability_bins(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> list[dict]:
    """Reliability table with the same bins as `ece`: [{lo, hi, n, conf, acc}], empty bins omitted."""
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            out.append({"lo": round(float(lo), 4), "hi": round(float(hi), 4), "n": int(m.sum()),
                        "conf": round(float(conf[m].mean()), 4), "acc": round(float(correct[m].mean()), 4)})
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def selective_risk(conf: np.ndarray, correct: np.ndarray, coverages=(0.1, 0.25, 0.5, 0.75, 1.0)) -> dict:
    if len(conf) == 0:
        return {}
    order = np.argsort(-conf)
    err = 1.0 - correct[order]
    return {f"risk@{c}": float(err[: max(1, int(round(c * len(err))))].mean()) for c in coverages}


def evaluate(logits: list[np.ndarray], decs: list[Decision], temps: Optional[dict] = None) -> dict:
    """Overall + per kind / per source / per language metrics. `temps` = {kind: temperature}."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(decs):
        groups["all"].append(i)
        groups[f"kind:{d.kind}"].append(i)
        groups[f"source:{d.source}"].append(i)
        groups[f"lang:{d.language}"].append(i)
    res = {}
    for name, idxs in groups.items():
        nll, brier, conf, corr, ev_p, ev_t = [], [], [], [], [], []
        for i in idxs:
            d = decs[i]
            T = (temps or {}).get(d.kind, 1.0)
            p = softmax(logits[i], T)
            t = np.array(d.target)
            nll.append(float(-(t * np.log(np.clip(p, 1e-12, 1))).sum()))
            brier.append(float(((p - t) ** 2).sum()))
            conf.append(float(p.max()))
            corr.append(float(top1_correct(p, t)))
            if d.kind == "score":
                v = np.array(d.option_values if d.option_values is not None else range(len(t)), dtype=float)
                ev_p.append(float((p * v).sum()))
                ev_t.append(float((t * v).sum()))
        conf_a, corr_a = np.array(conf), np.array(corr)
        r = {"n": len(idxs), "acc": float(corr_a.mean()), "nll": float(np.mean(nll)),
             "brier": float(np.mean(brier)), "ece": ece(conf_a, corr_a), "mean_conf": float(conf_a.mean())}
        if ev_p:
            r["score_mae"] = float(np.mean(np.abs(np.array(ev_p) - np.array(ev_t))))
            r["score_spearman"] = spearman(np.array(ev_p), np.array(ev_t))
        if name == "all" or name.startswith("kind:"):
            r.update(selective_risk(conf_a, corr_a))
            r["reliability"] = reliability_bins(conf_a, corr_a)
        res[name] = r
    return res


def fit_temperature(logits: list[np.ndarray], targets: list[np.ndarray], iters: int = 200) -> float:
    """Minimise soft-target NLL over a single temperature (LBFGS on log T)."""
    import torch

    if not logits:
        return 1.0
    K = max(len(x) for x in logits)
    L = torch.full((len(logits), K), -1e9, dtype=torch.float64)
    Tg = torch.zeros((len(logits), K), dtype=torch.float64)
    for i, (l, t) in enumerate(zip(logits, targets)):
        L[i, : len(l)] = torch.tensor(l, dtype=torch.float64)
        Tg[i, : len(t)] = torch.tensor(t, dtype=torch.float64)
    logT = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.5, max_iter=iters)

    def closure():
        opt.zero_grad()
        loss = -(Tg * torch.log_softmax(L / logT.exp(), dim=-1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(logT.detach().exp().clamp(0.05, 20.0))
