import gzip
import json

from mimir_decide.di_overlap import flag, request_texts, scan, train_index
from mimir_decide.di_table import benchmark_table, length_stats, summary_table
from mimir_decide.schema import write_parquet

from conftest import make_decision as mk


def test_overlap_flags_exact_and_ngram_matches_but_not_unrelated_requests(tmp_path):
    shared = " ".join(f"w{i}" for i in range(40))
    write_parquet([mk(id="t1", group_id="g1", state=shared), mk(id="t2", group_id="g2", state="something entirely different here")],
                  tmp_path / "train.parquet")
    hashes, fps = train_index(tmp_path / "train.parquet")
    rows = [
        {"_evaluation": {"dataset": "A"}, "state": "", "questions": {"q": {"instructions": shared}}},                  # exact
        {"_evaluation": {"dataset": "A"}, "state": "", "questions": {"q": {"instructions": shared + " plus tail"}}},   # n-grams
        {"_evaluation": {"dataset": "B"}, "state": {"x": 1}, "questions": {"q": {"instructions": "unrelated words only"}}},
    ]
    p = tmp_path / "rows.jsonl.gz"
    with gzip.open(p, "wt") as f:
        f.write("\n".join(json.dumps(r) for r in rows))
    stats = scan(p, hashes, fps)
    assert stats["A"] == {"requests": 2, "exact": 1, "ngram": 2, "either": 2} and stats["B"]["either"] == 0
    assert request_texts(rows[2])[0].startswith('{"x": 1}')
    assert flag("unrelated words only", hashes, fps) == (False, False)


def test_tables_read_scores_tolerantly_and_length_stats_count_statuses(tmp_path):
    for name, idx in (("a", 40.0), ("b", 55.5)):
        d = tmp_path / name
        d.mkdir()
        (d / "scores.json").write_text(json.dumps({"index": idx, "raw_index": idx + 20,
            "areas": {"tools": {"skill": idx / 2, "coverage": 0.9}, "language": {"skill": idx, "coverage": 1.0}},
            "benchmarks": {"1": {"skill": 10.0, "coverage": 1.0}, "24": {"skill": 30.0, "coverage": 0.5}}}))
        rows = [{"dataset": "MMLU", "status": "ok", "raw_output": {"max_tokens": 5000}},
                {"dataset": "MMLU", "status": "unsupported", "raw_output": None},
                {"dataset": "BFCL", "status": "ok", "raw_output": {"max_tokens": 300}}]
        with gzip.open(d / "results.jsonl.gz", "wt") as f:
            f.write("\n".join(json.dumps(r) for r in rows))
    t = summary_table([tmp_path / "a", tmp_path / "b"], ["A", "B"])
    assert "| A | 40.0 | 60.0 |" in t and "55.5" in t and "language skill" in t
    assert "| 24 | 30.0 (50.0%) | 30.0 (50.0%) |" in benchmark_table([tmp_path / "a", tmp_path / "b"], ["A", "B"])
    st = length_stats(tmp_path / "a")
    assert st["status"] == {"ok": 2, "unsupported": 1} and st["longer_than"]["4096"] == 1 and st["longer_than"]["8192"] == 0
    assert st["unsupported_by_benchmark"] == {"MMLU": 1}
