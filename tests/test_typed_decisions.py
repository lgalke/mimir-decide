import json

import numpy as np
import pytest

from conftest import make_decision as mk
from mimir_decide import typed_decisions as td


def test_decision_metrics_match_hand_computation():
    g = np.array([0.5, 0.25, 0.25])
    p = np.array([0.25, 0.5, 0.25])
    m = td.decision_metrics(p, g)
    assert m["acc"] == 0.0 and m["conf"] == 0.5
    assert m["kl"] == pytest.approx(0.5 * np.log(0.5 / 0.25) + 0.25 * np.log(0.25 / 0.5) + 0.25 * np.log(1))
    assert m["brier"] == pytest.approx(0.25 ** 2 + 0.25 ** 2)                     # sum over options of squared error
    assert m["nll"] == pytest.approx(-(0.5 * np.log(0.25) + 0.25 * np.log(0.5) + 0.25 * np.log(0.25)))
    assert td.decision_metrics(g, g)["kl"] == pytest.approx(0.0) and td.decision_metrics(g, g)["acc"] == 1.0


def test_zero_probability_is_clipped_not_infinite():
    m = td.decision_metrics(np.array([1.0, 0.0]), np.array([0.5, 0.5]))
    assert np.isfinite(m["kl"]) and 10 < m["kl"] < 20                                  # large but finite, like the README's Jev row


def test_uniform_kl_and_brier_follow_the_published_definitions():
    # KL(gold || uniform) = log K - H(gold); Brier = sum (1/K - gold)^2. These reproduce the published Uniform row on the real data.
    d = mk(options=["a", "b", "c", "d"], target=[0.7, 0.1, 0.1, 0.1])
    r = td.references([d], [d])["uniform"]["overall"]
    g = np.array(d.target)
    assert r["kl_from_gold"] == pytest.approx(np.log(4) + np.sum(g * np.log(g)))
    assert r["brier"] == pytest.approx(np.sum((0.25 - g) ** 2))


def test_option_keys_and_grouping():
    ch = mk(id="te_x_1:action", source="localllama/customer_service", kind="choice",
            options=["continue: Let it run.", "stop: Halt."], target=[0.4, 0.6])
    sc = mk(id="te_x_1:risk", source="localllama/customer_service", kind="score", options=["low", "high"], target=[0.4, 0.6])
    nl = mk(id="te_x_1:needs_review", source="localllama/customer_service", kind="noul", options=["yes!", "no!"], target=[0.4, 0.6])
    assert td.option_keys(ch) == ["continue", "stop"] and td.option_keys(sc) == ["0", "1"] and td.option_keys(nl) == ["true", "false"]
    res = td.score([np.array(d.target) for d in (ch, sc, nl)], [ch, sc, nl])
    assert set(res["by_kind"]) == {"choice", "score", "noul"} and set(res["by_workflow"]) == {"customer_service"}
    assert "customer_service/risk" in res["by_question"] and res["overall"]["accuracy"] == 1.0


def test_temperature_softmax_and_refusal_without_final():
    assert td.softmax_T(np.array([1.0, 2.0, 3.0]), 1.0).sum() == pytest.approx(1.0)
    flat = td.softmax_T(np.array([1.0, 2.0]), 100.0)
    assert abs(flat[0] - flat[1]) < 0.01                                          # high temperature flattens
    with pytest.raises(SystemExit):
        td.main(["--run_dir", "x"])                                                # no --final: the test split is not read


@pytest.mark.network  # real benchmark data and tokenizer; a tiny random model stands in for the base model
def test_scoring_run_writes_results_and_predictions_and_logs_the_access(tmp_path):
    import yaml
    from transformers.models.hrm_text.configuration_hrm_text import HrmTextConfig
    from transformers.models.hrm_text.modeling_hrm_text import HrmTextForCausalLM

    from mimir_decide import zeroshot

    rev = "521b40b36a79918014544b970d4c2669ff1530eb"
    cfg = HrmTextConfig(vocab_size=262144, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                        num_attention_heads=4, num_key_value_heads=4, head_dim=16, H_cycles=2, L_cycles=3,
                        L_bp_cycles=[3, 3], max_position_embeddings=2048, prefix_lm=True, pad_token_id=0,
                        bos_token_id=2, eos_token_id=106, initializer_range=0.0255)
    HrmTextForCausalLM(cfg).save_pretrained(tmp_path / "base")
    conf = tmp_path / "letter.yaml"
    conf.write_text(yaml.safe_dump({"model_type": "letter", "base_model": str(tmp_path / "base"), "revision": None,
                                    "tokenizer": "danish-foundation-models/DFM-Mimir-v1.5", "tokenizer_revision": rev,
                                    "max_len": 2048}))
    run = tmp_path / "run"
    zeroshot.main(["--config", str(conf), "--run_dir", str(run)])
    td.main(["--run_dir", str(run), "--checkpoint", "zero-shot", "--final", "--limit", "30", "--batch_size", "8"])
    res = json.loads((run / "eval" / "typed_decisions_test_uncal.json").read_text())      # no calibration.json: raw probabilities
    assert res["overall"]["n"] == 30 and res["calibrated"] is False and "scorer" in res
    assert not (run / "eval" / "typed_decisions_test.json").exists()
    (run / "calibration.json").write_text(json.dumps({"temperatures": {"noul": 2.0, "choice": 2.0, "score": 2.0}}))
    td.main(["--run_dir", str(run), "--checkpoint", "zero-shot", "--final", "--limit", "30", "--batch_size", "8"])
    cal = json.loads((run / "eval" / "typed_decisions_test.json").read_text())             # one call writes both versions
    assert cal["calibrated"] is True and cal["overall"]["accuracy"] == res["overall"]["accuracy"]   # temperature keeps the argmax
    assert cal["overall"]["mean_conf"] < res["overall"]["mean_conf"]                                # and flattens the probabilities
    preds = json.loads((run / "eval" / "typed_decisions_test_uncal_predictions.json").read_text())
    assert len(preds) == 30 and abs(sum(preds[0]["probs"]) - 1.0) < 1e-4 and len(preds[0]["keys"]) == len(preds[0]["probs"])
    assert "typed-decisions benchmark test" in (run / "eval" / "test_access_log.jsonl").read_text()
    # the loader returns the benchmark's size
    assert len(td.load_decisions("test")) == 2000
