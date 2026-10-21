from mimir_decide.mixture_report import dominant_script, report

from conftest import make_decision as mk


def test_report_numbers_on_a_small_planted_mixture():
    decs = [mk(id=f"a{i}", source="d/a", state=f"state {i}", options=["x", "y"], target=[1.0, 0.0]) for i in range(6)]
    decs += [mk(id=f"b{i}", source="d/b", kind="noul", options=["yes", "no"], target=[0.5, 0.5], soft=True,
                state=f"noul {i}") for i in range(2)]
    r = report(decs)
    assert r["decisions"] == 8 and r["sources"] == 2 and r["top1_share"] == 0.75
    assert r["kind_share"] == {"choice": 0.75, "noul": 0.25, "score": 0.0}
    assert r["binary_share"] == 1.0 and r["soft_share"] == 0.25
    assert r["mean_target_entropy_bits"] == 0.25         # 6 x 0 bits, 2 x 1 bit over 8 decisions
    assert 1.0 < r["effective_sources"] < 2.0


def test_script_mix_detects_writing_systems():
    assert dominant_script("The quick brown fox") == "LATIN"
    assert dominant_script("Привет мир, как дела") == "CYRILLIC"
    assert dominant_script("日本語のテキスト") in ("CJK", "HIRAGANA", "KATAKANA")
    assert dominant_script("12345 !!!") == "NONE"
