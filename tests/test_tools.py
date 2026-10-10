import numpy as np

from mimir_decide.compare import pool, split_by_seen
from mimir_decide.inference import unpermute


def test_unpermute_maps_presented_order_back_to_original():
    perm = [2, 0, 1]                      # presented option j is original option perm[j]
    presented = np.array([30.0, 10.0, 20.0])  # logits in presented order
    assert unpermute(presented, perm).tolist() == [10.0, 20.0, 30.0]
    assert unpermute(np.array([1.0, 2.0]), [0, 1]).tolist() == [1.0, 2.0]


def test_compare_pools_seen_and_unflagged_sources_by_n():
    res = {"source:a": {"n": 10, "acc": 0.9, "nll": 0.2, "brier": 0.1},
           "source:b": {"n": 30, "acc": 0.5, "nll": 1.0, "brier": 0.5},
           "source:c": {"n": 10, "acc": 0.3, "nll": 1.5, "brier": 0.7}}
    ev = {"results": res, "sources_seen_by_mimir": ["a", "b"]}
    s = split_by_seen(ev)
    assert s["seen_by_mimir"]["n"] == 40 and abs(s["seen_by_mimir"]["acc"] - (9 + 15) / 40) < 1e-9
    assert s["not_flagged"] == {"n": 10, "acc": 0.3, "nll": 1.5, "brier": 0.7}
    assert pool(res, ["zzz"]) == {"n": 0}


def test_compare_explains_a_missing_eval_file(tmp_path):
    import json

    import pytest

    from mimir_decide.compare import main

    run = tmp_path / "runs" / "letter-zeroshot"
    (run / "eval").mkdir(parents=True)
    (run / "eval" / "validation_uncal.json").write_text(json.dumps({}))
    with pytest.raises(SystemExit) as e:
        main([str(run), "--file", "validation.json"])
    msg = str(e.value)
    assert "validation.json does not exist" in msg and "validation_uncal.json" in msg and "evaluate" in msg
