import json

from mimir_decide.convert import bekko, helpsteer2, localllama, massive, tasksource
from mimir_decide.convert.common import Counter0, norm_split


def test_norm_split():
    assert norm_split("dev") == "validation" and norm_split("test") == "test" and norm_split(None) is None


def test_tasksource_kinds_and_split_alias():
    dr = Counter0()
    base = dict(state="s", id="a:train:1", group_id="a:train:1", source="a", variant="direct", license="mit",
                license_use="commercial", question_id="decision", question="q?")
    ch = tasksource.row_to_decision({**base, "kind": "choice", "options": ["x", "y"], "target": [0, 1], "split": "train"},
                                    "train", dr)
    assert ch and ch.kind == "choice" and ch.target == [0.0, 1.0] and not ch.soft
    no = tasksource.row_to_decision({**base, "kind": "noul", "options": [], "target": [0.7], "split": "train"}, "train", dr)
    assert no and no.options == ["yes", "no"] and abs(no.target[0] - 0.7) < 1e-9 and no.soft
    sc = tasksource.row_to_decision({**base, "kind": "score", "options": ["1", "2", "3"], "target": [0, 0, 1], "split": "train"},
                                    "train", dr)
    assert sc and sc.option_values == [0.0, 1.0, 2.0]
    # HF validation split rows carry split == "dev"; must be accepted as validation, rejected as train
    ok = tasksource.row_to_decision({**base, "kind": "choice", "options": ["x", "y"], "target": [1, 0], "split": "dev"},
                                    "validation", dr)
    assert ok and ok.split == "validation"
    bad = tasksource.row_to_decision({**base, "kind": "choice", "options": ["x", "y"], "target": [1, 0], "split": "test"},
                                     "train", dr)
    assert bad is None and dr["tasksource_split_column_mismatch"] == 1


def test_localllama_row():
    row = {
        "id": "tr_x_000", "workflow": "wf", "split": "train", "state": json.dumps({"task": "t", "n": 1}),
        "questions": json.dumps({
            "action": {"type": "choice", "instructions": "what?", "criteria": {"go": "Proceed.", "stop": "Halt."}},
            "risk": {"type": "score", "instructions": "risk?", "criteria": ["low", "mid", "high"]},
            "needs_review": {"type": "noul", "instructions": "review?", "criteria": {"true": "yes!", "false": "no!"}},
            "duplicate": {"type": "noul", "instructions": "dup?"},
        }),
        "gold": json.dumps({
            "action": {"probabilities": {"go": 0.25, "stop": 0.75}},
            "risk": {"probabilities": {"0": 0.5, "1": 0.25, "2": 0.25}},
            "needs_review": {"probabilities": {"true": 0.9, "false": 0.1}},
            "duplicate": {"probabilities": {"true": 0.2, "false": 0.8}},
        }),
    }
    ds = {d.id.split(":")[1]: d for d in localllama.row_to_decisions(row, "train", Counter0())}
    assert set(ds) == {"action", "risk", "needs_review", "duplicate"}
    assert ds["action"].options[1].startswith("stop") and ds["action"].target == [0.25, 0.75]
    assert ds["risk"].kind == "score" and ds["risk"].option_values == [0.0, 1.0, 2.0]
    assert ds["needs_review"].options == ["yes!", "no!"] and ds["duplicate"].options == ["yes", "no"]
    assert "task: t" in ds["action"].state and all(d.license == "apache-2.0" for d in ds.values())


def test_helpsteer2_ids_unique_across_upstream_splits():
    row = {"prompt": "p", "response": "r", "helpfulness": 3, "correctness": 2, "coherence": 4, "complexity": 1, "verbosity": 0}
    a = {d.id for d in helpsteer2.row_to_decisions(row, 0, "train", Counter0(), "train")}
    b = {d.id for d in helpsteer2.row_to_decisions(row, 0, "test", Counter0(), "validation")}
    assert len(a) == 5 and not a & b  # regression: ids used to collide (index restarts per upstream file)


def test_massive_options_contain_gold_and_are_deterministic():
    rows = [{"id": str(i), "partition": "dev", "intent": "alarm_set", "utt": f"u{i}"} for i in range(5)]
    intents = ["alarm_set", "alarm_query", "weather_query", "music_play", "iot_hue_lightoff", "qa_stock", "news_query"]
    mk = lambda: list(massive.rows_to_decisions(rows, "da-DK", "validation", intents, 4, 0, Counter0()))
    a, b = mk(), mk()
    assert [d.options for d in a] == [d.options for d in b]
    for d in a:
        assert len(d.options) == 4 and d.options.count("alarm: set") == 1
        assert d.target[d.options.index("alarm: set")] == 1.0 and d.split == "validation" and d.language == "da"
    assert massive.humanise_intent("iot_hue_lightoff") == "iot: hue lightoff"
    assert list(massive.rows_to_decisions(rows, "da-DK", "train", intents, 4, 0, Counter0())) == []  # partition filter


def _bekko_row(decisions, targets):
    return {"input": {"state_json": json.dumps({"q": "hi"}), "decisions": decisions}, "targets": targets,
            "split": "train", "case_id": "c1", "group_id": "g1", "language": "en", "input_hash": "h"}


def test_bekko_choice_score_ranking_and_unsupported():
    crit = lambda i, d, v=None: {"id": i, "description_json": json.dumps(d), "value": v}
    row = _bekko_row(
        [
            {"id": "intent", "kind": "judgment", "type": "choice", "instructions_json": json.dumps("pick"),
             "criteria": [crit("a", "Alpha"), crit("b", "Beta")], "documents": []},
            {"id": "qual", "kind": "judgment", "type": "score", "instructions_json": json.dumps("rate"),
             "criteria": [crit("hi", "High", 2.0), crit("lo", "Low", 0.0), crit("mid", "Mid", 1.0)], "documents": []},
            {"id": "rank", "kind": "ranking", "type": None, "instructions_json": json.dumps("rank"),
             "criteria": [], "documents": [{"id": "0", "content_json": json.dumps("doc zero")},
                                           {"id": "1", "content_json": json.dumps("doc one")}]},
            {"id": "weird", "kind": "judgment", "type": "other", "instructions_json": json.dumps("?"),
             "criteria": [crit("x", "X")], "documents": []},
        ],
        [
            {"decision_id": "intent", "ids": ["a", "b"], "probabilities": [0.2, 0.8]},
            {"decision_id": "qual", "ids": ["hi", "lo", "mid"], "probabilities": [0.1, 0.6, 0.3]},
            {"decision_id": "rank", "ids": ["0", "1"], "probabilities": [0.0, 1.0]},
            {"decision_id": "weird", "ids": ["x"], "probabilities": [1.0]},
        ],
    )
    dr = Counter0()
    ds = {d.id.split(":")[-1]: d for d in bekko.row_to_decisions(row, "sub", "mit", "train", dr)}
    assert set(ds) == {"intent", "qual", "rank"} and dr["bekko_unsupported_type:other"] == 1
    assert ds["intent"].options == ["Alpha", "Beta"] and ds["intent"].target == [0.2, 0.8]  # order + soft target kept
    assert ds["qual"].options == ["Low", "Mid", "High"] and ds["qual"].target == [0.6, 0.3, 0.1]  # sorted by value
    assert ds["qual"].option_values == [0.0, 1.0, 2.0]
    assert ds["rank"].options == ["doc zero", "doc one"] and ds["rank"].target == [0.0, 1.0]


def test_bekko_subset_license():
    up = lambda lic, status: {"license": lic, "license_review_status": status}
    src = {
        "good": {"train_and_validation": {"upstream_datasets": [up(["apache-2.0"], "verified")]}},
        "qual": {"train_and_validation": {"upstream_datasets": [up(["apache-2.0"], "qualified")]}},
        "unk": {"train_and_validation": {"upstream_datasets": [up("unknown", "verified")]}},
        "nc": {"train_and_validation": {"upstream_datasets": [up(["cc-by-nc-4.0"], "verified")]}},
        "mixed": {"train_and_validation": {"upstream_datasets": [up(["mit"], "verified"), up(["cc-by-nc-4.0"], "verified")]}},
        "none": {"train_and_validation": {"upstream_datasets": []}},
    }
    r = {k: bekko.subset_license(k, src)[:2] for k in [*src, "missing"]}
    assert r["good"] == (True, "ok") and r["qual"] == (False, "license_qualified")
    assert r["unk"][0] is False and r["nc"] == (False, "non_commercial") and r["mixed"][0] is False
    assert r["none"] == (False, "no_upstream_license_info") and r["missing"] == (False, "no_source_entry")
    assert bekko.subset_license("qual", src, accept_qualified=True)[0] is True
