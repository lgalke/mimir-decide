"""Mixture v1 options of the builder: nested sizes, caps, floors, frozen evaluation, exclusions, long-state guard."""
import pytest

from mimir_decide import build_mixture as bm
from mimir_decide.schema import read_parquet

from conftest import make_decision as mk


def _cfg(tmp_path, name="out", **extra):
    cfg = {
        "seed": 0, "output_dir": str(tmp_path / name), "max_state_chars": 2000, "max_options": 5,
        "per_source_cap": 1000, "read_limit_per_source": 100000, "max_total_train": 10000, "sampling_alpha": 0.5,
        "eval_caps": {"validation_per_source": 100, "test_per_source": 100, "heldout_per_source": 100},
        "held_out_sources": [], "sources": {"A": {"enabled": True}},
    }
    cfg.update(extra)
    return cfg


def _patch(monkeypatch, data):
    def fake(name, split, cfg, drops, limit, excl, config=None):
        from dataclasses import replace
        return iter([replace(d) for d in data.get((name, split), [])])
    monkeypatch.setattr(bm, "iter_source", fake)


def _pool():
    train = []
    for src, n, kind, opts in (("A/big", 600, "choice", list("abc")), ("A/bin", 400, "choice", ["x", "y"]),
                               ("A/noul", 300, "noul", ["yes", "no"]), ("A/score", 300, "score", list("01234")),
                               ("proc/p1", 300, "choice", list("abcd")), ("proc/p2", 300, "choice", list("abcd"))):
        for i in range(n):
            t = [0.0] * len(opts)
            t[i % len(opts)] = 1.0
            train.append(mk(id=f"{src}:{i}", group_id=f"g:{src}:{i}", source=src, kind=kind, options=opts, target=t,
                            state=f"unique state {src} number {i} " + "w" * (i % 7), dataset="A",
                            option_values=[float(j) for j in range(len(opts))] if kind == "score" else None))
    return train


SAMPLING = {"alpha": 0.5, "total": 1200, "max_source_share": 0.3, "families": {"procedural": "proc/*"},
            "family_caps": {"procedural": 0.15}, "kind_floors": {"noul": 0.10, "score": 0.10},
            "max_binary_share": 0.5, "sizes": {"small": 400, "mid": 800}}


def test_v1_sampling_nested_sizes_caps_and_floors(tmp_path, monkeypatch):
    _patch(monkeypatch, {("A", "train"): _pool(), ("A", "validation"): [], ("A", "test"): []})
    m = bm.build(_cfg(tmp_path, sampling=SAMPLING))
    out = tmp_path / "out"
    full, mid, small = (read_parquet(out / f) for f in ("train.parquet", "train_mid.parquet", "train_small.parquet"))
    assert len(full) <= 1200 and len(mid) <= 800 and len(small) <= 400 and len(small) < len(mid) < len(full)
    ids = [{d.id for d in x} for x in (small, mid, full)]
    assert ids[0] <= ids[1] <= ids[2]                                  # nested
    proc = sum(1 for d in full if d.source.startswith("proc/"))
    assert proc <= 0.15 * 1200 + 5                                      # family cap
    assert max(sum(1 for d in full if d.source == s) for s in {d.source for d in full}) <= 0.3 * 1200 + 1
    for recs in (full, mid):
        assert sum(d.kind == "noul" for d in recs) / len(recs) >= 0.10 - 0.03
        assert sum(d.kind == "score" for d in recs) / len(recs) >= 0.10 - 0.03
    assert m["subsets"]["small"]["decisions"] == len(small) and m["sampling"]["converged"]


def test_v1_is_deterministic(tmp_path, monkeypatch):
    _patch(monkeypatch, {("A", "train"): _pool(), ("A", "validation"): [], ("A", "test"): []})
    bm.build(_cfg(tmp_path, "o1", sampling=SAMPLING))
    bm.build(_cfg(tmp_path, "o2", sampling=SAMPLING))
    a = sorted(d.id for d in read_parquet(tmp_path / "o1" / "train.parquet"))
    b = sorted(d.id for d in read_parquet(tmp_path / "o2" / "train.parquet"))
    assert a == b


def test_frozen_eval_must_equal_the_reference(tmp_path, monkeypatch):
    val = [mk(id=f"v{i}", group_id=f"gv{i}", split="validation", state=f"val {i}") for i in range(5)]
    data = {("A", "train"): _pool(), ("A", "validation"): val, ("A", "test"): []}
    _patch(monkeypatch, data)
    bm.build(_cfg(tmp_path, "ref"))
    m = bm.build(_cfg(tmp_path, "same", frozen_eval_from=str(tmp_path / "ref")))
    assert all(m["frozen_eval_identical_to_reference"].values())
    data[("A", "validation")] = val[:4]                                 # a different eval set
    _patch(monkeypatch, data)
    with pytest.raises(AssertionError, match="frozen reference"):
        bm.build(_cfg(tmp_path, "diff", frozen_eval_from=str(tmp_path / "ref")))


def test_excluded_sources_never_enter_train(tmp_path, monkeypatch):
    (tmp_path / "extra.yaml").write_text("held_out_sources:\n- proc/p1\ntask_novel_sources:\n- A/bin\n")
    _patch(monkeypatch, {("A", "train"): _pool(), ("A", "validation"): [], ("A", "test"): []})
    m = bm.build(_cfg(tmp_path, exclude_sources_from=[str(tmp_path / "extra.yaml")], exclude_sources=["A/noul"]))
    srcs = {d.source for d in read_parquet(tmp_path / "out" / "train.parquet")}
    assert not srcs & {"proc/p1", "A/bin", "A/noul"} and "proc/p2" in srcs
    assert m["drops"]["A"]["excluded_source"] == 400 + 300 + 300


def test_annotation_stripping_applies_to_train_only_and_is_reported(tmp_path, monkeypatch):
    pool = [mk(id=f"m{i}", group_id=f"gm{i}", source="A/mit", state=f"mit state {i}", license="mit, mit (original source)")
            for i in range(5)]
    ev = [mk(id="ev", group_id="gev", source="A/mit", split="validation", state="eval of mit source",
             license="mit, mit (original source)")]
    _patch(monkeypatch, {("A", "train"): pool, ("A", "validation"): ev, ("A", "test"): []})
    m0 = bm.build(_cfg(tmp_path, "strict"))
    assert m0["counts"]["train"] == 0 and m0["drops"]["A"]["license:not_on_allowlist"] == 5
    m1 = bm.build(_cfg(tmp_path, "lenient", license_strip_annotations=True))
    assert m1["counts"]["train"] == 5 and m1["newly_admitted_by_annotation_stripping"] == 5
    assert m1["counts"]["validation"] + m1["counts"]["calib"] == 0     # the eval pass keeps the strict parser (frozen)
    assert "A/mit" in (tmp_path / "lenient" / "reports" / "license_newly_admitted.csv").read_text()


def test_long_eval_state_still_guards_train_when_enabled(tmp_path, monkeypatch):
    shared = " ".join(f"w{i}" for i in range(60))
    long_eval = mk(id="ev", group_id="gev", split="validation", source="A/ev", state=shared + " " + "pad " * 600)
    leaky = mk(id="tr", group_id="gtr", source="A/other", state=shared + " tail")
    ok = mk(id="ok", group_id="gok", source="A/other", state="a completely different short state")
    data = {("A", "train"): [leaky, ok], ("A", "validation"): [long_eval], ("A", "test"): []}
    _patch(monkeypatch, data)
    kw = dict(max_state_chars=500, train_max_state_chars=5000, eval_max_state_chars=500)
    off = bm.build(_cfg(tmp_path, "off", **kw))
    on = bm.build(_cfg(tmp_path, "on", guard_long_eval_states=True, **kw))
    assert off["counts"]["train"] == 2                                  # v0 behaviour: the long eval state is not hashed
    assert on["counts"]["train"] == 1 and on["drops"]["A"]["leak_ngram_in_eval"] == 1


def test_infeasible_floor_is_reported_not_hidden(tmp_path, monkeypatch, capsys):
    pool = [mk(id=f"c{i}", group_id=f"gc{i}", source="A/c", state=f"choice state {i}") for i in range(500)]
    pool += [mk(id=f"n{i}", group_id=f"gn{i}", source="A/n", kind="noul", options=["yes", "no"], target=[1.0, 0.0],
                state=f"noul state {i}") for i in range(10)]
    _patch(monkeypatch, {("A", "train"): pool, ("A", "validation"): [], ("A", "test"): []})
    m = bm.build(_cfg(tmp_path, sampling={"total": 400, "kind_floors": {"noul": 0.5}}))
    assert m["sampling"]["converged"] is False and "WARNING" in capsys.readouterr().out
