"""E12 tooling: select held-out sources with unseen label sets, derive a mixture without them, compare per source."""
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

from conftest import make_decision as mk
from mimir_decide import derive_mixture, select_heldout
from mimir_decide.compare import main as compare_main
from mimir_decide.schema import write_parquet

LABELS = {
    "S/unique1": ["alpha", "beta", "gamma", "delta", "epsilon"],
    "S/unique2": ["red tone", "blue tone", "green tone"],
    "S/unique3": ["north", "south", "east", "west"],
    "S/shared_a": ["entailment", "neutral", "contradiction"],   # same label set as shared_b -> seen elsewhere
    "S/shared_b": ["entailment", "neutral", "contradiction"],
    "S/vocab": ["alpha", "zzz"],                                 # option strings overlap with unique1
}


def _recs(source, split, n, start=0):
    opts = LABELS.get(source)
    out = []
    for i in range(n):
        o = opts if opts else [f"answer {source} {start + i} {j}" for j in range(3)]  # per-item options
        out.append(mk(id=f"{source}:{split}:{start + i}", group_id=f"{source}:{split}:{start + i}", source=source,
                      split=split, state=f"state {source} {split} {start + i}", options=list(o),
                      target=[1.0] + [0.0] * (len(o) - 1)))
    return out


@pytest.fixture
def mixture(tmp_path):
    d = tmp_path / "mix"
    (d / "eval" / "test").mkdir(parents=True)
    (d / "heldout_tasks").mkdir()
    (d / "reports").mkdir()
    sources = [*LABELS, "S/peritem"]
    write_parquet([r for s in sources for r in _recs(s, "train", 120)], d / "train.parquet")
    write_parquet([r for s in sources for r in _recs(s, "validation", 80)], d / "validation.parquet")
    write_parquet([r for s in sources for r in _recs(s, "validation", 40, 1000)], d / "calib.parquet")
    write_parquet([r for s in sources for r in _recs(s, "test", 50)], d / "eval" / "test" / "test.parquet")
    write_parquet(_recs("M/old-heldout", "validation", 7), d / "heldout_tasks" / "heldout.parquet")
    (d / "mixture_manifest.json").write_text(json.dumps({"counts": {"train": 0}}))
    flagged = [s for s in sources if s != "S/unique3"]
    (d / "reports" / "mimir_overlap.json").write_text(json.dumps({"sources_seen_by_mimir": flagged}))
    return d


def test_scores_and_eligibility(mixture):
    scores = select_heldout.score_sources(select_heldout.source_stats(mixture))
    assert scores["S/shared_a"]["seen_fraction"] == 1.0 and scores["S/unique1"]["seen_fraction"] == 0.0
    assert scores["S/peritem"]["repeat_fraction"] == 0.0
    assert scores["S/vocab"]["string_overlap"] == 0.5 and scores["S/unique1"]["string_overlap"] == pytest.approx(0.2)
    ok, why = select_heldout.eligible(scores)
    assert ok == ["S/unique1", "S/unique2", "S/unique3"]
    assert why["no fixed label set (per-item options)"] == 1 and why["option set used by another source"] == 2
    assert why["option strings used by other sources"] == 1


def test_selection_is_seeded_and_separates_flagged_sources(mixture, tmp_path):
    out = tmp_path / "heldout.yaml"
    select_heldout.main(["--data_dir", str(mixture), "--out", str(out), "--n_flagged", "1", "--n_unflagged", "1", "--seed", "3"])
    chosen = yaml.safe_load(out.read_text())["held_out_sources"]
    assert "S/unique3" in chosen and len(chosen) == 2 and set(chosen) <= {"S/unique1", "S/unique2", "S/unique3"}
    rep = json.loads(out.with_suffix(".report.json").read_text())
    assert rep["chosen"]["S/unique3"]["flagged_by_name"] is False and rep["n_eligible"] == 3
    out2 = tmp_path / "again.yaml"
    select_heldout.main(["--data_dir", str(mixture), "--out", str(out2), "--n_flagged", "1", "--n_unflagged", "1", "--seed", "3"])
    assert yaml.safe_load(out2.read_text()) == yaml.safe_load(out.read_text())     # same seed, same choice


def test_derive_mixture_moves_eval_rows_and_removes_train_rows(mixture, tmp_path):
    dst = tmp_path / "derived"
    held = ["S/unique1", "S/unique3"]
    counts = derive_mixture.derive(mixture, dst, held)
    assert counts["train_removed"] == 2 * 120 and counts["validation_moved"] == 2 * 80
    for f in ["train.parquet", "validation.parquet", "calib.parquet", "eval/test/test.parquet"]:
        assert not set(pq.read_table(dst / f, columns=["source"])["source"].to_pylist()) & set(held), f
    heldout = pq.read_table(dst / "heldout_tasks" / "heldout.parquet").to_pylist()
    by_src = {}
    for r in heldout:
        by_src.setdefault(r["source"], []).append(r["split"])
    assert len(by_src["M/old-heldout"]) == 7                                          # old held-out rows kept
    assert len(by_src["S/unique1"]) == 80 + 40 + 50 and set(by_src["S/unique1"]) == {"validation", "test"}
    assert not any(r["split"] == "train" for r in heldout)
    m = json.loads((dst / "mixture_manifest.json").read_text())
    assert m["derived"]["extra_held_out_sources"] == held and (dst / "reports" / "mimir_overlap.json").exists()
    with pytest.raises(AssertionError):                                               # destination must be empty
        derive_mixture.derive(mixture, dst, held)
    with pytest.raises(AssertionError):                                               # unknown source
        derive_mixture.derive(mixture, tmp_path / "other", ["S/does-not-exist"])


def _run(path: Path, accs: dict, conf=0.8):
    (path / "eval").mkdir(parents=True)
    res = {f"source:{s}": {"n": 100, "acc": a, "nll": 0.5, "brier": 0.3, "mean_conf": conf, "ece": 0.02} for s, a in accs.items()}
    (path / "eval" / "heldout_unseen.json").write_text(json.dumps({"results": res, "calibrated": True, "n_skipped_too_long": 0}))


def test_compare_prints_pooled_gain_and_per_source_table(tmp_path, capsys):
    _run(tmp_path / "zero", {"S/a": 0.5, "S/b": 0.6})
    _run(tmp_path / "slot", {"S/a": 0.7, "S/b": 0.8})
    (tmp_path / "src.yaml").write_text(yaml.safe_dump({"held_out_sources": ["S/a", "S/b"]}))
    compare_main([str(tmp_path / "zero"), str(tmp_path / "slot"), "--file", "heldout_unseen.json",
                  "--sources-file", str(tmp_path / "src.yaml")])
    out = capsys.readouterr().out
    assert "gain=+20.0" in out and "S/a" in out and "0.700" in out and "pooled over 2 sources" in out
