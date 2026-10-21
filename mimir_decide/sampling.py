"""Quota and selection logic for the diverse mixture (D31): pure functions, no I/O.

  solve_quotas   per-source quotas  q_s = min(n_s, c * n_s**alpha, max_source), c found by bisection so that the
                 total is reached; families (for example procedural generators, bekko) can be capped as a share.
  priorities     a deterministic uniform number per decision id; the order inside a source is fixed by it.
  solve_boosts   weights per kind and for binary choice/score decisions that move the composition towards floors
                 (noul, score) and a cap (binary share), by simulation with the real selection rule.
  select         top-q decisions of one source by key log(u)/w (weighted sampling without replacement); because the
                 order is fixed, selecting a larger q (or a larger total) only adds decisions: subsets are nested.
"""
from __future__ import annotations

import fnmatch
import hashlib
import math
from dataclasses import dataclass

import numpy as np

KIND_CODE = {"choice": 0, "noul": 1, "score": 2}


def family_of(source: str, dataset: str, patterns: dict[str, str] | None) -> str:
    """Family of a source: first matching fnmatch pattern of `patterns` {family: pattern}, else its dataset."""
    for fam, pat in (patterns or {}).items():
        if fnmatch.fnmatch(source, pat):
            return fam
    return dataset


def _quota(n: int, c: float, alpha: float, smax: int | None) -> float:
    q = min(float(n), c * n ** alpha)
    return min(q, float(smax)) if smax is not None else q


def _solve_c(ns: list[int], target: float, alpha: float, smax: int | None) -> float:
    """Bisection for c with sum(q_s(c)) = target; inf when even the full pool is below the target."""
    if not ns:
        return 0.0
    if sum(min(n, smax) if smax is not None else n for n in ns) <= target:
        return math.inf
    lo, hi = 0.0, float(max(ns)) ** (1 - alpha) + 1.0
    for _ in range(80):
        c = (lo + hi) / 2
        if sum(_quota(n, c, alpha, smax) for n in ns) < target:
            lo = c
        else:
            hi = c
    return lo


def solve_quotas(counts: dict[str, int], total: int, alpha: float = 0.5, max_source: int | None = None,
                 families: dict[str, str] | None = None, family_caps: dict[str, float] | None = None) -> dict[str, int]:
    """Per-source quotas that sum to about `total` (less when the pool is smaller or the caps bind).

    `families` maps source -> family; `family_caps` maps family -> maximal share of `total`. A capped family gets its
    own constant c_f, the remaining families share what is left (a few rounds, fixed point)."""
    fam = families or {}
    caps = family_caps or {}
    clamped: dict[str, dict[str, int]] = {}
    out: dict[str, int] = {}
    for _ in range(len(caps) + 2):
        free = [s for s in counts if fam.get(s, s) not in clamped]
        left = total - sum(sum(q.values()) for q in clamped.values())
        c = _solve_c([counts[s] for s in free], max(0, left), alpha, max_source)
        out = {s: int(_quota(counts[s], c, alpha, max_source)) for s in free}
        changed = False
        for f in sorted({fam.get(s, s) for s in free}):
            members = [s for s in free if fam.get(s, s) == f]
            if f in caps and sum(out[s] for s in members) > caps[f] * total:
                cf = _solve_c([counts[s] for s in members], caps[f] * total, alpha, max_source)
                clamped[f] = {s: int(_quota(counts[s], cf, alpha, max_source)) for s in members}
                changed = True
        if not changed:
            break
    for q in clamped.values():
        out.update(q)
    return out


def priorities(ids: list[str], seed: int) -> np.ndarray:
    """Deterministic uniform numbers in (0, 1) per decision id."""
    u = np.empty(len(ids))
    for i, x in enumerate(ids):
        h = hashlib.blake2b(f"{seed}:{x}".encode(), digest_size=8).digest()
        u[i] = (int.from_bytes(h, "big") + 0.5) / 2 ** 64
    return u


@dataclass
class SourceArrays:
    u: np.ndarray        # priority numbers
    kind: np.ndarray     # int8 codes (KIND_CODE)
    binary: np.ndarray   # bool: exactly two options


def select(src: SourceArrays, q: int, boosts: dict[str, float], bin_pen: float) -> np.ndarray:
    """Indices of the q decisions with the largest key log(u)/w, where w = boost(kind) * bin_pen for binary
    choice/score decisions. Stable, so equal keys keep their order; nested for growing q."""
    if q <= 0:
        return np.empty(0, dtype=np.int64)
    w = np.array([boosts["choice"], boosts["noul"], boosts["score"]])[src.kind]
    w = np.where(src.binary & (src.kind != KIND_CODE["noul"]), w * bin_pen, w)
    key = np.log(src.u) / w
    order = np.argsort(-key, kind="stable")
    return order[: min(q, len(order))]


def composition(sources: dict[str, SourceArrays], quotas: dict[str, int], boosts: dict[str, float],
                bin_pen: float) -> dict[str, float]:
    n = noul = score = binary = 0
    for s, a in sources.items():
        idx = select(a, quotas.get(s, 0), boosts, bin_pen)
        n += len(idx)
        noul += int((a.kind[idx] == KIND_CODE["noul"]).sum())
        score += int((a.kind[idx] == KIND_CODE["score"]).sum())
        binary += int(a.binary[idx].sum())
    n = max(n, 1)
    return {"n": n, "noul": noul / n, "score": score / n, "choice": 1 - (noul + score) / n, "binary": binary / n}


def solve_boosts(sources: dict[str, SourceArrays], quotas: dict[str, int], floors: dict[str, float] | None = None,
                 max_binary: float | None = None, iters: int = 40, tol: float = 0.004):
    """Find boosts for noul/score (floors) and the penalty for binary choice/score decisions (cap) by simulation.
    Boosts never go below 1 and the penalty never above 1, so a target that is already met changes nothing.
    Returns (boosts, bin_pen, composition, converged)."""
    floors = floors or {}
    boosts = {"choice": 1.0, "noul": 1.0, "score": 1.0}
    bin_pen = 1.0
    comp = composition(sources, quotas, boosts, bin_pen)
    for _ in range(iters):
        ok = True
        for k, fl in floors.items():
            if comp[k] < fl - tol:
                boosts[k] = min(200.0, boosts[k] * (fl / max(comp[k], 1e-4)) ** 0.8)
                ok = False
            elif boosts[k] > 1.0 and comp[k] > fl + tol:
                boosts[k] = max(1.0, boosts[k] * (fl / comp[k]) ** 0.8)
                ok = False
        if max_binary is not None:
            if comp["binary"] > max_binary + tol:
                bin_pen = max(0.01, bin_pen * (max_binary / comp["binary"]) ** 0.8)
                ok = False
            elif bin_pen < 1.0 and comp["binary"] < max_binary - tol:
                bin_pen = min(1.0, bin_pen * (max_binary / max(comp["binary"], 1e-4)) ** 0.8)
                ok = False
        if ok:
            return boosts, bin_pen, comp, True
        comp = composition(sources, quotas, boosts, bin_pen)
    ok = all(comp[k] >= fl - 2 * tol for k, fl in floors.items()) and (max_binary is None or comp["binary"] <= max_binary + 2 * tol)
    return boosts, bin_pen, comp, ok
