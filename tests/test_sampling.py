import numpy as np

from mimir_decide import sampling as sp


def _sources(spec, seed=0):
    """spec: {name: (n, p_noul, p_score, p_binary_choice)}"""
    rng = np.random.default_rng(seed)
    out = {}
    for name, (n, pn, ps, pb) in spec.items():
        kind = rng.choice([0, 1, 2], size=n, p=[1 - pn - ps, pn, ps]).astype(np.int8)
        binary = (kind == 1) | (rng.random(n) < pb)
        out[name] = sp.SourceArrays(sp.priorities([f"{name}:{i}" for i in range(n)], 0), kind, binary)
    return out


def test_quotas_reach_total_and_respect_caps():
    counts = {"a": 100000, "b": 20000, "c": 500, "d": 50}
    q = sp.solve_quotas(counts, 12000, alpha=0.5, max_source=8000)
    assert q["d"] == 50 and q["c"] == 500                        # small sources are taken completely
    assert max(q.values()) <= 8000
    assert abs(sum(q.values()) - 12000) <= 4
    # the caps leave a pool of 16,550: asking for more returns that pool and no more
    assert sum(sp.solve_quotas(counts, 30000, alpha=0.5, max_source=8000).values()) == 16550
    assert all(q[s] <= counts[s] for s in counts)


def test_quotas_return_the_whole_pool_when_it_is_smaller_than_the_total():
    counts = {"a": 10, "b": 20}
    assert sp.solve_quotas(counts, 1000) == counts


def test_temperature_flattens_large_sources():
    counts = {"big": 1_000_000, "small": 1000}
    q = sp.solve_quotas(counts, 20000, alpha=0.5)
    assert q["big"] / q["small"] < 100     # a proportional split would give 1000:1


def test_family_cap_limits_a_family_and_the_rest_fills_up():
    counts = {f"proc/{i}": 50000 for i in range(10)} | {"x": 400000, "y": 400000}
    fam = {s: ("procedural" if s.startswith("proc/") else s) for s in counts}
    q = sp.solve_quotas(counts, 100000, alpha=0.5, families=fam, family_caps={"procedural": 0.10})
    proc = sum(v for s, v in q.items() if s.startswith("proc/"))
    assert proc <= 0.10 * 100000 + 10
    assert abs(sum(q.values()) - 100000) <= 20          # the other sources take up the rest


def test_family_of_uses_patterns_then_dataset():
    pats = {"procedural": "tasksource/procedural-typed-decisions/*"}
    assert sp.family_of("tasksource/procedural-typed-decisions/arithmetic", "tasksource", pats) == "procedural"
    assert sp.family_of("tasksource/boolq", "tasksource", pats) == "tasksource"


def test_selection_is_nested_and_deterministic():
    a = _sources({"s": (5000, 0.2, 0.1, 0.3)})["s"]
    b = {"choice": 1.0, "noul": 3.0, "score": 2.0}
    small, big = sp.select(a, 500, b, 0.5), sp.select(a, 2000, b, 0.5)
    assert set(small) <= set(big) and len(set(big)) == 2000
    assert (sp.select(a, 500, b, 0.5) == small).all()


def test_boosts_reach_the_floors_and_cap():
    srcs = _sources({"a": (40000, 0.05, 0.03, 0.6), "b": (40000, 0.10, 0.05, 0.5), "c": (20000, 0.30, 0.10, 0.4)})
    quotas = {s: 8000 for s in srcs}
    base = sp.composition(srcs, quotas, {"choice": 1, "noul": 1, "score": 1}, 1.0)
    assert base["noul"] < 0.20 and base["score"] < 0.15
    boosts, pen, comp, ok = sp.solve_boosts(srcs, quotas, {"noul": 0.20, "score": 0.15}, max_binary=0.60)
    assert ok and comp["noul"] >= 0.20 - 0.01 and comp["score"] >= 0.15 - 0.01 and comp["binary"] <= 0.60 + 0.01
    assert boosts["noul"] > 1 and boosts["score"] > 1


def test_boosts_change_nothing_when_targets_already_hold():
    srcs = _sources({"a": (20000, 0.4, 0.3, 0.1)})
    boosts, pen, comp, ok = sp.solve_boosts(srcs, {"a": 5000}, {"noul": 0.2, "score": 0.15}, max_binary=0.9)
    assert ok and boosts == {"choice": 1.0, "noul": 1.0, "score": 1.0} and pen == 1.0
