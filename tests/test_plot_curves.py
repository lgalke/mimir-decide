import json

from mimir_decide.plot_curves import read_evals


def test_reads_jsonl_run_dirs_and_text_logs_with_python_dicts(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "train_log.jsonl").write_text(
        json.dumps({"step": 10, "loss": 1.0}) + "\n" + json.dumps({"eval": {"step": 1000, "val_nll": 0.6, "val_acc": 0.77, "val_ece": 0.03, "val_n": 2000}}) + "\n")
    (tmp_path / "x.log").write_text("noise\n{'step': 1000, 'val_nll': 0.62, 'val_acc': 0.74, 'val_ece': 0.02, 'val_n': 2000}\n"
                                    "{'step': 20, 'loss': 0.1}\n{'step': 2000, 'val_nll': 0.6, 'val_acc': 0.77, 'val_ece': 0.04, 'val_n': 2000}\n")
    assert [r["step"] for r in read_evals(d)] == [1000] and read_evals(d)[0]["val_acc"] == 0.77
    assert [r["step"] for r in read_evals(tmp_path / "x.log")] == [1000, 2000]
