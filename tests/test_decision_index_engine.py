import math

import pytest
import torch

from mimir_decide.decision_index_engine import MimirDecideEngine, Unsupported, to_decisions
from mimir_decide.formatting import Formatter


class FakeModel(torch.nn.Module):
    """Logit of option j is j: the last option always wins."""
    def forward(self, batch):
        v = batch["valid"]
        return torch.arange(v.size(1), dtype=torch.float32).expand(v.size(0), -1).masked_fill(~v, -1e9)


def engine(tok, mode="slot", max_len=2048, temps=None):
    e = object.__new__(MimirDecideEngine)
    e.model, e.tok, e.device, e.batch_size, e.mode, e.max_len, e.temps = FakeModel(), tok, "cpu", 4, mode, max_len, temps
    e.meta = {"model_type": mode, "max_len": max_len, "marker_id": 5}
    e.run = type("R", (), {"name": "fake"})()
    e.fmt = Formatter(tok, mode, 10 ** 9, 5)
    return e


QS = {"q1": {"type": "choice", "instructions": "Which colour?", "criteria": {"r": "red", "g": "green", "b": "blue"}},
      "q2": {"type": "noul", "instructions": {"claim": "x is true"}}}


def test_request_is_mapped_to_decisions():
    keys, decs, ok = to_decisions({"a": 1}, QS)
    assert keys == ["q1", "q2"] and ok == [["r", "g", "b"], ["yes", "no"]]
    assert decs[0].kind == "choice" and decs[0].options == ["red", "green", "blue"]
    assert decs[1].kind == "noul" and decs[1].options == ["yes", "no"]
    assert decs[0].state == '{"a":1}' and decs[1].question == '{"claim":"x is true"}'
    with pytest.raises(Unsupported):
        to_decisions("s", {"q": {"type": "score", "criteria": {}}})


@pytest.mark.parametrize("mode", ["slot", "letter"])
def test_answers_are_valid_distributions(tok, mode):
    resp, raw = engine(tok, mode)("some state", QS)
    a = resp["answers"]
    assert a["q1"]["choice"] == "b" and set(a["q1"]["probabilities"]) == {"r", "g", "b"}
    assert math.isclose(sum(a["q1"]["probabilities"].values()), 1.0, abs_tol=1e-9)
    assert a["q2"]["type"] == "noul" and 0 < a["q2"]["noul"] < 0.5   # logits [0, 1] for [yes, no]


def test_temperature_changes_confidence_not_choice(tok):
    p1 = engine(tok)("s", QS)[0]["answers"]["q1"]
    p2 = engine(tok, temps={"choice": 2.0, "noul": 1.0, "score": 1.0})("s", QS)[0]["answers"]["q1"]
    assert p1["choice"] == p2["choice"] and p2["probabilities"]["b"] < p1["probabilities"]["b"]


def test_no_truncation_requests_that_do_not_fit_are_refused(tok):
    long_state = "word " * 400
    with pytest.raises(Unsupported):
        engine(tok, max_len=256)(long_state, QS)
    assert engine(tok, max_len=4096)(long_state, QS)[0]["answers"]["q1"]["choice"] == "b"


def test_letter_model_refuses_more_than_26_options(tok):
    q = {"q": {"type": "choice", "instructions": "?", "criteria": {str(i): f"option {i}" for i in range(27)}}}
    with pytest.raises(Unsupported):
        engine(tok, "letter")("s", q)
    assert engine(tok, "slot")("s", q)[0]["answers"]["q"]["choice"] == "26"   # the slot model has no such limit


def test_overlong_option_is_refused_not_cut(tok):
    q = {"q": {"type": "choice", "instructions": "?", "criteria": {"a": "x " * 200, "b": "y"}}}
    with pytest.raises(Unsupported):
        engine(tok)("s", q)
