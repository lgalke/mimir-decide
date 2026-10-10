"""E12 option A: evaluation-only extra tasks, licence limits, exclusion of training sources, leakage and label novelty."""
import json

import pyarrow.parquet as pq
import pytest
import yaml

from conftest import make_decision as mk
from mimir_decide import build_extra_eval as bx
from mimir_decide.evaluate import PATHS
from mimir_decide.schema import write_parquet

WORDS = [f"w{i}x" for i in range(400)]


def passage(k, n=80):
    return " ".join(WORDS[k * 60 : k * 60 + n])         # distinct word windows -> distinct 12-gram fingerprints


def rows(source, n, labels, license="cc-by-4.0", state=lambda i: None, split="validation"):
    return [mk(id=f"{source}:{split}:{i}", group_id=f"{source}:{i}", source=source, split=split, license=license,
               state=state(i) or f"unique state of {source} number {i}", options=list(labels),
               target=[1.0] + [0.0] * (len(labels) - 1)) for i in range(n)]


@pytest.fixture
def pilot(tmp_path):
    d = tmp_path / "pilot"
    d.mkdir()
    train = rows("T/a", 120, ["pos", "neg"], split="train")
    train += rows("T/b", 5, ["x1", "x2", "x3"], split="train", state=lambda i: passage(i))   # 5 long train passages
    write_parquet(train, d / "train.parquet")
    return d


def run(pilot, tmp_path, cand):
    return bx.build({}, pilot, tmp_path / "extra", tmp_path / "cfg" / "sources.yaml", rows=cand, min_eval=50)


def test_chosen_and_rejected_sources(pilot, tmp_path):
    novel = ["alpha", "beta", "gamma", "delta", "epsilon"]
    cand = (rows("X/ok", 80, novel)
            + rows("X/unlicensed", 80, ["q1", "q2", "q3"], license="unspecified")
            + rows("X/noncommercial", 80, ["r1", "r2", "r3"], license="cc-by-nc-4.0")
            + rows("T/a", 80, ["pos", "neg"])                                # a training source
            + rows("X/same_labels", 80, ["pos", "neg"])                      # option set used by a training source
            + rows("X/few", 10, ["n1", "n2", "n3"]))                         # too few rows
    rep = run(pilot, tmp_path, cand)
    assert list(rep["chosen"]) == ["X/ok"] and rep["rows"] == 80
    assert rep["row_drops"]["licence:unknown_license"] == 80 and rep["row_drops"]["licence:non_commercial"] == 80
    assert rep["row_drops"]["source_in_training_mixture"] == 80
    assert "option set used by a training source" in rep["rejected"]["X/same_labels"]["reason"]
    assert "fewer eval rows" in rep["rejected"]["X/few"]["reason"]
    assert yaml.safe_load((tmp_path / "cfg" / "sources.yaml").read_text())["held_out_sources"] == ["X/ok"]
    out = pq.read_table(tmp_path / "extra" / "extra_eval" / "extra_eval.parquet").to_pylist()
    assert {r["source"] for r in out} == {"X/ok"} and len(out) == 80
    assert (tmp_path / "extra" / "reports" / "extra_eval_report.json").exists()


def test_rows_copied_from_training_are_dropped_even_when_rendered_differently(pilot, tmp_path):
    novel = ["alpha", "beta", "gamma", "delta", "epsilon"]
    leaked_exact = lambda i: passage(i) if i < 3 else None                       # 3 rows equal a train state
    leaked_ngram = lambda i: f"reformatted as: {passage(3 + i - 10)} (end)" if 10 <= i < 12 else None   # 2 re-rendered
    cand = rows("X/leaky", 70, novel, state=lambda i: leaked_exact(i) or leaked_ngram(i))
    rep = run(pilot, tmp_path, cand)
    assert rep["row_drops"]["leak_state_in_train"] == 3 and rep["row_drops"]["leak_ngram_in_train"] == 2
    assert rep["rows"] == 65 and list(rep["chosen"]) == ["X/leaky"]
    kept = {r["state"] for r in pq.read_table(tmp_path / "extra" / "extra_eval" / "extra_eval.parquet").to_pylist()}
    assert passage(0) not in kept and not any(s.startswith("reformatted as") for s in kept)


def test_extra_split_is_known_to_evaluate_and_per_item_options_are_rejected(pilot, tmp_path):
    assert PATHS["extra"] == "extra_eval/extra_eval.parquet"
    per_item = [mk(id=f"P:{i}", group_id=f"P:{i}", source="X/qa", license="mit", split="test", state=f"question {i}",
                   options=[f"answer {i} {j}" for j in range(3)], target=[1.0, 0.0, 0.0]) for i in range(80)]
    rep = run(pilot, tmp_path, per_item)
    assert rep["chosen"] == {} and "no fixed label set" in rep["rejected"]["X/qa"]["reason"]


def test_audit_reads_the_extra_file(tmp_path):
    import inspect

    from mimir_decide import audit_mimir_overlap
    assert "extra_eval/extra_eval.parquet" in inspect.getsource(audit_mimir_overlap.main)


def test_familiar_task_is_reported_and_split_into_the_task_novel_list(pilot, tmp_path):
    from mimir_decide.build_extra_eval import familiar_task

    assert familiar_task("bekko/snli", {"tasksource/snli", "x/y"}) == "tasksource/snli"
    assert familiar_task("bekko/logical_entailment", {"tasksource/logical-entailment"}) == "tasksource/logical-entailment"
    assert familiar_task("bekko/hans", {"tasksource/snli"}) is None
    assert familiar_task("bekko/ab", {"tasksource/ab"}) is None                  # too short to match by name
    t = rows("tasksource/newsnli", 60, ["old a", "old b", "old c"], split="train")      # a training source with that task name
    from mimir_decide.schema import write_parquet
    write_parquet(rows("T/a", 120, ["pos", "neg"], split="train") + t, pilot / "train.parquet")
    cand = rows("bekko/newsnli", 80, ["entails the claim", "contradicts the claim", "unrelated to the claim"]) \
        + rows("bekko/brandnew", 80, ["north", "south", "east", "west"])
    rep = run(pilot, tmp_path, cand)
    assert set(rep["chosen"]) == {"bekko/newsnli", "bekko/brandnew"}
    assert rep["chosen"]["bekko/newsnli"]["familiar_task_in_training"] == "tasksource/newsnli"
    assert rep["chosen"]["bekko/brandnew"]["familiar_task_in_training"] is None and rep["n_task_novel_sources"] == 1
    y = yaml.safe_load((tmp_path / "cfg" / "sources.yaml").read_text())
    assert y["held_out_sources"] == ["bekko/brandnew", "bekko/newsnli"] and y["task_novel_sources"] == ["bekko/brandnew"]
