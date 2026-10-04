"""The mixture builder must never let eval content (any split of any source) into train."""
import pytest

from mimir_decide import build_mixture as bm
from mimir_decide.schema import read_parquet, state_hash

from conftest import make_decision as mk


def _cfg(tmp_path, held=()):
    return {
        "seed": 0, "output_dir": str(tmp_path / "out"), "max_state_chars": 1000, "max_options": 3,
        "per_source_cap": 100, "read_limit_per_source": 100, "max_total_train": 1000, "sampling_alpha": 0.5,
        "eval_caps": {"validation_per_source": 100, "test_per_source": 100, "heldout_per_source": 100},
        "held_out_sources": list(held),
        "sources": {"A": {"enabled": True}, "B": {"enabled": True}},
    }


def _patch(monkeypatch, data):
    """data[(name, split)] -> list of Decision"""
    def fake(name, split, cfg, drops, limit, excl):
        return iter([_copy(d) for d in data.get((name, split), [])])
    monkeypatch.setattr(bm, "iter_source", fake)


def _copy(d):
    from dataclasses import replace
    return replace(d)


def test_leakage_license_holdout_dedup_and_final_check(tmp_path, monkeypatch):
    test_state = "the shared secret test passage"
    data = {
        ("A", "train"): [
            mk(id="a1", group_id="ga1", source="A/x", state="clean train one"),
            mk(id="a2", group_id="ga2", source="A/x", state="clean train two"),
            mk(id="a3", group_id="ga3", source="A/x", state=test_state),                 # same state as B's TEST row
            mk(id="a4", group_id="gval", source="A/x", state="different text, same group as a val row"),
            mk(id="a5", group_id="ga5", source="A/x", state="clean train one"),          # exact duplicate of a1
            mk(id="a6", group_id="ga6", source="A/nc", state="nc licensed", license="cc-by-nc-4.0"),
            mk(id="a7", group_id="ga7", source="A/unk", state="unknown licence", license="unspecified"),
            mk(id="a8", group_id="ga8", source="A/held", state="held out task train row"),
            mk(id="a9", group_id="ga9", source="A/x", state="  The SHARED   secret test passage "),  # normalised match
            mk(id="a10", group_id="ga10", source="A/x", state="s" * 5000),               # too long
            mk(id="a11", group_id="ga11", source="A/x", state="many options", options=list("abcdef"),
               target=[0, 1, 0, 0, 0, 0]),                                               # option subsampling keeps gold
        ],
        ("A", "validation"): [mk(id="av1", group_id="gval", source="A/x", split="validation", state="val state")],
        ("A", "test"): [],
        ("B", "train"): [mk(id="b1", group_id="gb1", dataset="B", source="B/y", state="b clean")],
        ("B", "validation"): [mk(id="bv1", group_id="gbv", dataset="B", source="B/held", split="validation",
                                 state="held val")],
        ("B", "test"): [
            mk(id="bt1", group_id="gbt", dataset="B", source="B/y", split="test", state=test_state),
            mk(id="bt2", group_id="gbt2", dataset="B", source="B/nc", split="test", state="nc test state",
               license="cc-by-nc-4.0"),  # eval from NC source: hashed for the guard, never written
        ],
    }
    _patch(monkeypatch, data)
    m = bm.build(_cfg(tmp_path, held=["A/held", "B/held"]))
    out = tmp_path / "out"
    train = read_parquet(out / "train.parquet")
    ids = {d.id for d in train}
    assert ids == {"a1", "a2", "a11", "b1"}, ids
    d11 = next(d for d in train if d.id == "a11")
    assert len(d11.options) == 3 and d11.options[d11.target.index(1.0)] == "b"  # gold kept
    dr = m["drops"]["A"]
    assert dr["leak_state_in_eval"] == 2 and dr["leak_group_in_eval"] == 1 and dr["duplicate"] == 1
    assert dr["held_out_task"] == 1 and dr["state_too_long"] == 1
    assert dr["license:non_commercial"] == 1 and dr["license:unknown_license"] == 1
    # independent re-check of the written files
    eval_files = [out / "validation.parquet", out / "calib.parquet", out / "heldout_tasks" / "heldout.parquet",
                  out / "eval" / "test" / "test.parquet"]
    ev = [d for f in eval_files for d in read_parquet(f)]
    assert {state_hash(d) for d in train}.isdisjoint({state_hash(d) for d in ev})
    assert "bt2" not in {d.id for d in ev}                      # NC eval row never written ...
    held = read_parquet(out / "heldout_tasks" / "heldout.parquet")
    assert {d.id for d in held} == {"bv1"}                      # held-out validation goes to heldout, not validation
    assert all(d.split == "train" for d in train)


def test_nc_eval_state_still_blocks_train(tmp_path, monkeypatch):
    data = {
        ("A", "train"): [mk(id="a1", group_id="g1", source="A/x", state="nc test state")],
        ("A", "validation"): [], ("A", "test"): [
            mk(id="t1", group_id="gt", source="A/x", split="test", state="nc test state", license="cc-by-nc-4.0")],
        ("B", "train"): [], ("B", "validation"): [], ("B", "test"): [],
    }
    _patch(monkeypatch, data)
    m = bm.build(_cfg(tmp_path))
    assert m["counts"]["train"] == 0 and m["drops"]["A"]["leak_state_in_eval"] == 1


def test_validation_and_calibration_are_disjoint_by_group(tmp_path, monkeypatch):
    val = [mk(id=f"v{i}", group_id=f"g{i // 2}", source="A/x", split="validation", state=f"val {i}") for i in range(40)]
    _patch(monkeypatch, {("A", "validation"): val})
    bm.build(_cfg(tmp_path))
    out = tmp_path / "out"
    v, c = read_parquet(out / "validation.parquet"), read_parquet(out / "calib.parquet")
    assert v and c and {d.group_id for d in v}.isdisjoint({d.group_id for d in c})


def test_temperature_quotas():
    q = bm.temperature_quotas({"big": 100000, "mid": 1000, "small": 10}, 20000, 0.5)
    assert q["small"] == 10 and q["mid"] <= 1000 and abs(sum(q.values()) - 20000) < 50
    assert q["big"] / q["mid"] < 100000 / 1000  # flattened
    assert bm.temperature_quotas({"a": 5}, 100, 0.5) == {"a": 5}


def test_cap_options_drops_when_too_many_positive():
    d = mk(options=list("abcdef"), target=[1 / 6] * 6)
    assert bm.cap_options(d, 3) is None


def test_ngram_guard_catches_rerendered_content_but_not_boilerplate(tmp_path, monkeypatch):
    """Same underlying example rendered differently by another collection must not leak into train
    (regression: tasksource contains HelpSteer2, rendered differently from our helpsteer2 converter)."""
    body = ("the committee reviewed every submitted proposal carefully and decided that funding should be "
            "allocated according to the published criteria without exception during the following quarter")
    header = ("You are an expert reviewer please read the following material and answer the question that "
              "follows after the material has been presented in full detail")
    eval_rows = [mk(id=f"e{i}", group_id=f"ge{i}", source="A/x", split="test",
                    state=f"{header} item {i} " + " ".join(f"w{i}x{j}" for j in range(30))) for i in range(8)]
    eval_rows.append(mk(id="ehs", group_id="geh", source="A/hs", split="test",
                        state=f"Prompt:\nsome prompt\n\nResponse:\n{body}"))
    train_rows = [
        mk(id="t_leak", group_id="gtl", source="B/hs", state=f"response_text: {body} (re-rendered)"),   # leak
        mk(id="t_clean", group_id="gtc", source="A/x",                                                    # same header only
           state=f"{header} item 99 " + " ".join(f"z{j}" for j in range(30))),
        mk(id="t_short", group_id="gts", source="A/x", state="short unrelated state"),
    ]
    _patch(monkeypatch, {("A", "test"): eval_rows, ("B", "train"): train_rows[:1], ("A", "train"): train_rows[1:]})
    cfg = _cfg(tmp_path)
    m = bm.build(cfg)
    kept = {d.id for d in read_parquet(tmp_path / "out" / "train.parquet")}
    assert kept == {"t_clean", "t_short"}, kept
    assert m["drops"]["B"]["leak_ngram_in_eval"] == 1
    assert m["eval_hash_counts"]["boilerplate_ngrams_ignored"] > 0   # the shared header was recognised as template


def test_fingerprints_basic():
    from mimir_decide.schema import fingerprints
    assert fingerprints("too short to fingerprint") == []
    a = " ".join(f"word{i}" for i in range(60))
    assert fingerprints(a) == fingerprints("  " + a.upper() + "\n") and len(fingerprints(a)) > 3
    assert not set(fingerprints(a)) & set(fingerprints(" ".join(f"other{i}" for i in range(60))))


def test_ngram_guard_trusts_upstream_split_within_a_source(tmp_path, monkeypatch):
    """Templated/synthetic sources share scenario text between their own train and test by design; only
    other sources' eval content counts as leakage (regression: 1345 false drops on LocalLLaMA)."""
    scenario = ("the agent was asked to delete personal data for the accounts in the erasure queue while "
                "respecting every named constraint and then reported its trace summary afterwards")
    ev = mk(id="e1", group_id="ge1", source="A/synth", split="test", state=f"{scenario} run 1 steps 7 errors 1")
    same_src = mk(id="t1", group_id="gt1", source="A/synth", state=f"{scenario} run 2 steps 9 errors 0")
    other_src = mk(id="t2", group_id="gt2", source="B/other", dataset="B", state=f"{scenario} run 3 steps 4")
    _patch(monkeypatch, {("A", "test"): [ev], ("A", "train"): [same_src], ("B", "train"): [other_src]})
    m = bm.build(_cfg(tmp_path))
    kept = {d.id for d in read_parquet(tmp_path / "out" / "train.parquet")}
    assert kept == {"t1"}, kept
    ex = [__import__("json").loads(l) for l in open(tmp_path / "out" / "reports" / "ngram_leak_examples.jsonl")]
    assert len(ex) == 1 and ex[0]["train_source"] == "B/other" and ex[0]["matched_eval_source"] == "A/synth"


def test_unmatched_heldout_pattern_is_reported(tmp_path, monkeypatch):
    _patch(monkeypatch, {("A", "train"): [mk(id="a", group_id="g", source="A/x")]})
    m = bm.build(_cfg(tmp_path, held=["nothing/*"]))
    assert m["heldout_patterns_unmatched"] == ["nothing/*"]
