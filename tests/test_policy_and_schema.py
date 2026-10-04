import pytest

from mimir_decide import licenses
from mimir_decide.schema import normalise_target, norm_text, text_hash, state_hash


@pytest.mark.parametrize("lic,use,ok,why", [
    ("apache-2.0", "commercial", True, "ok"),
    ("cc-by-4.0, CC BY 4.0 (DPI)", "commercial", True, "ok"),
    ("apache-2.0, Apache License 2.0 (DPI)", None, True, "ok"),
    ("cc-by-sa-4.0", None, True, "share_alike"),
    ("cc-by-nc-4.0", "non-commercial", False, "non_commercial"),
    ("CC BY-NC-SA 4.0 (DPI)", None, False, "non_commercial"),
    ("cc-by-4.0", "non-commercial", False, "non_commercial"),
    ("unspecified", None, False, "unknown_license"),
    ("other", None, False, "unknown_license"),
    ("", None, False, "unknown_license"),
    ("gpl-3.0", None, False, "not_on_allowlist"),
    ("cc-by-4.0, gpl-3.0", None, False, "not_on_allowlist"),  # compound: every part must pass
])
def test_license_policy(lic, use, ok, why):
    assert licenses.check(lic, use) == (ok, why)


def test_validate(mk):
    assert mk().validate() is None
    assert mk(options=["a"], target=[1.0]).validate() == "lt2_options"
    assert mk(target=[0.5, 0.2, 0.2]).validate() == "target_not_normalised"
    assert mk(target=[1.0, 0.0]).validate() == "target_len_mismatch"
    assert mk(split="dev").validate() == "bad_split"
    assert mk(kind="rank").validate() == "bad_kind"
    assert mk(state="  ").validate() == "empty_text"
    assert mk(target=[float("nan"), 0.5, 0.5]).validate() == "bad_target_value"


def test_normalise_target():
    assert normalise_target([2, 2]) == [0.5, 0.5]
    assert normalise_target([0, 0]) is None
    assert normalise_target([-1, 1]) == [0.0, 1.0]


def test_hash_normalisation():
    assert text_hash("Hello   World\n") == text_hash("hello world")
    assert text_hash("ǅ") == text_hash("ǆ")  # NFKC + lowercase
    assert text_hash("a") != text_hash("b")


def test_audit_flags_tasksource_collection_and_aliases():
    from mimir_decide.audit_mimir_overlap import audit
    sources = {"tasksource/boolq": "tasksource", "tasksource/HelpSteer2/coherence": "tasksource",
               "helpsteer2/coherence": "helpsteer2", "massive/da-DK": "massive", "localllama/customer_service": "localllama"}
    rep = audit(sources, ["tasksource__", "flan__", "dfm11-koolbardi-da__"])
    flagged = set(rep["sources_seen_by_mimir"])
    assert {"tasksource/boolq", "tasksource/HelpSteer2/coherence", "helpsteer2/coherence"} <= flagged
    assert "massive/da-DK" not in flagged and "localllama/customer_service" not in flagged
