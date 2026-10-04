"""docs/ must stay a valid, indexed OKF bundle. If this fails: `python -m mimir_decide.okf --write`, then fix the rest."""
import shutil
from pathlib import Path

from mimir_decide import okf


def test_docs_bundle_is_valid_and_indexes_are_current():
    assert okf.problems() == []


def test_every_decision_question_experiment_has_required_fields():
    for folder, status_values in {"design": {"accepted", "proposed", "superseded"},
                                  "questions": {"open", "answered", "dropped"},
                                  "experiments": {"planned", "running", "done", "abandoned"}}.items():
        for p in okf.concept_files(okf.DOCS / folder):
            fm = okf.frontmatter(p)
            if fm["type"] in {"Decision", "Question", "Experiment"}:
                assert fm.get("status") in status_values, f"{p.name}: bad status {fm.get('status')}"
                assert fm.get("title") and fm.get("description"), p.name


def test_ids_are_unique_and_sequential():
    for folder, prefix in {"design": "d", "questions": "q", "experiments": "e", "observations": "o"}.items():
        ids = [p.name.split("-")[0] for p in okf.concept_files(okf.DOCS / folder) if p.name[0] == prefix and p.name[1].isdigit()]
        assert ids == sorted(set(ids)), folder
        assert [int(i[1:]) for i in ids] == list(range(1, len(ids) + 1)), f"{folder}: ids not sequential {ids}"


def _copy_docs(tmp_path) -> Path:
    dst = tmp_path / "docs"
    shutil.copytree(okf.DOCS, dst)
    return dst


def test_checker_detects_missing_type_stale_index_and_broken_link(tmp_path):
    docs = _copy_docs(tmp_path)
    assert okf.problems(docs) == []
    page = docs / "research" / "jev.md"
    page.write_text(page.read_text().replace("type: Product", "type: ''"))
    assert any("jev.md" in p and "empty type" in p for p in okf.problems(docs))
    page.write_text(page.read_text().replace("type: ''", "type: Product"))
    (docs / "research" / "newpage.md").write_text("---\ntype: Concept\ntitle: New\ndescription: d\n---\nbody\n")
    assert any("research/index.md" in p and "stale" in p for p in okf.problems(docs))
    okf.main(["--write", "--docs", str(docs)])
    assert okf.problems(docs) == []
    (docs / "research" / "newpage.md").write_text("---\ntype: Concept\ntitle: New\ndescription: d\n---\n[x](/nope.md)\n")
    assert any("broken link /nope.md" in p for p in okf.problems(docs))


def test_log_order_is_checked(tmp_path):
    docs = _copy_docs(tmp_path)
    log = docs / "log.md"
    log.write_text(log.read_text() + "\n## 2030-01-01\n* **Init**: out of order\n")
    assert any("log.md" in p for p in okf.problems(docs))
