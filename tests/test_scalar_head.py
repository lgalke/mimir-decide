"""Scalar per-option head (D33): formatting, row packing, sampled negatives, model equivariance, training step."""
import math
import random

import torch
from transformers.models.hrm_text.configuration_hrm_text import HrmTextConfig
from transformers.models.hrm_text.modeling_hrm_text import HrmTextForCausalLM

from conftest import FakeTok, make_decision as mk
from mimir_decide.formatting import Collator, Formatter, MAX_OPTION_TOKENS, mode_for, subsample_negatives
from mimir_decide.inference import predict_logits
from mimir_decide.model import OptionScorerModel, decision_loss


def fmt(max_len=200):
    return Formatter(FakeTok(), "scalar", max_len, 5)


def tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = HrmTextConfig(vocab_size=9000, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                        num_key_value_heads=2, head_dim=16, H_cycles=2, L_cycles=3, L_bp_cycles=[3, 3],
                        max_position_embeddings=512, prefix_lm=True, pad_token_id=0, bos_token_id=2, eos_token_id=1,
                        initializer_range=0.02)
    m = OptionScorerModel(HrmTextForCausalLM(cfg))
    torch.nn.init.normal_(m.head[1].weight, std=0.5)         # non-trivial scores
    return m.eval()


def test_mode_for():
    assert [mode_for(x) for x in ("slot", "letter", "scalar")] == ["slot", "letter", "scalar"]


def test_one_row_per_option_sharing_the_state_and_ending_with_the_option():
    d = mk(state="the state text", options=["red", "green", "blue"], target=[0, 1, 0])
    e = fmt().encode(d)
    assert len(e.rows) == 3 and e.n_options == 3
    common = min(len(r) for r in e.rows)
    assert len({tuple(r[: common - 6]) for r in e.rows}) == 1             # same head and state
    tok = FakeTok()
    for r, o in zip(e.rows, ["red", "green", "blue"]):
        assert r[-len(o):] == tok(o)["input_ids"]                           # the option is the end of the row
    assert fmt().encode(d, [2, 1, 0]).rows[0][-4:] == tok("blue")["input_ids"]   # presented order follows perm


def test_long_option_is_cut_at_the_limit_and_too_little_room_drops_the_decision():
    d = mk(options=["x" * 400, "y"], target=[1, 0])
    e = fmt(max_len=800).encode(d)
    assert len(e.rows[0]) <= 800 and len(e.rows[0]) - len(e.rows[1]) >= MAX_OPTION_TOKENS - 20
    assert fmt(max_len=100).encode(d) is None


def test_pack_rows_maps_rows_back_to_decisions_and_options():
    f = fmt()
    d1 = mk(id="a", options=["a", "b", "c"], target=[0, 0, 1])
    d2 = mk(id="b", options=["p", "q"], target=[1, 0], kind="noul")
    c = Collator(f, False)
    batch = c.pack([f.encode(d1), f.encode(d2)], [d1, d2])
    assert batch["input_ids"].shape[0] == 5 and batch["valid"].shape == (2, 3)
    assert batch["row_dec"].tolist() == [0, 0, 0, 1, 1] and batch["row_opt"].tolist() == [0, 1, 2, 0, 1]
    assert (batch["last_pos"] == batch["attention_mask"].sum(1) - 1).all()
    assert batch["target"][0].tolist() == [0, 0, 1] and not batch["valid"][1, 2]


def test_negative_sampling_keeps_every_positive_and_corrects_the_normaliser():
    d = mk(options=list("abcdefghij"), target=[0, 0.25, 0, 0, 0.75, 0, 0, 0, 0, 0])
    perm = list(range(10))
    d2, p2, off = subsample_negatives(d, perm, 4, random.Random(0))
    assert "b" in d2.options and "e" in d2.options and len(d2.options) == 4          # 2 positives + 2 negatives
    assert math.isclose(sum(d2.target), 1.0) and p2 == [0, 1, 2, 3]
    pos = [i for i, t in enumerate(d2.target) if t > 0]
    assert all(off[i] == 0 for i in pos) and all(math.isclose(off[i], math.log(8 / 2)) for i in range(4) if i not in pos)
    d3, _, off3 = subsample_negatives(d, perm, 20, random.Random(0))                 # nothing to sample
    assert len(d3.options) == 10 and set(off3) == {0.0}
    d4, _, _ = subsample_negatives(mk(options=list("abcd"), target=[.5, .5, 0, 0]), [0, 1, 2, 3], 2, random.Random(1))
    assert len(d4.options) == 3                                                       # at least one negative stays


def test_collator_subsamples_only_choice_decisions_in_training():
    f = fmt(400)
    c = Collator(f, True, max_train_options=3)
    ch = mk(id="c", options=list("abcdefg"), target=[1, 0, 0, 0, 0, 0, 0])
    sc = mk(id="s", kind="score", options=list("01234"), target=[0, 0, 1, 0, 0], option_values=[0., 1, 2, 3, 4])
    batch, decs, drop = c([(ch, 1), (sc, 2)])
    assert batch["valid"].sum(1).tolist() == [3, 5] and batch["logit_offset"][0].max() > 0
    assert batch["logit_offset"][1].abs().sum() == 0                                  # score bins are never sampled


def test_scores_do_not_depend_on_presentation_order_or_padding():
    m = tiny_model()
    f = fmt()
    d = mk(state="a state about colours", options=["red", "green", "blue", "yellow"], target=[0, 1, 0, 0])
    long_other = mk(id="other", state="word " * 30, options=["a", "b"], target=[1, 0])
    c = Collator(f, False)
    with torch.no_grad():
        base = m(c.pack([f.encode(d)], [d]))[0, :4]
        rev = m(c.pack([f.encode(d, [3, 2, 1, 0])], [d]))[0, :4]
        padded = m(c.pack([f.encode(long_other), f.encode(d)], [long_other, d]))[1, :4]
    assert torch.allclose(base, rev.flip(0), atol=1e-4)
    assert torch.allclose(base, padded, atol=1e-4)


def test_predict_logits_returns_original_option_order_for_the_scalar_model():
    m = tiny_model()
    decs = [mk(id=f"d{i}", state=f"state number {i}", options=["red", "green", "blue"], target=[0, 1, 0])
            for i in range(3)]
    meta = {"model_type": "scalar", "max_len": 200, "marker_id": 5}
    a, _ = predict_logits(m, FakeTok(), meta, decs, batch_size=2)
    b, _ = predict_logits(m, FakeTok(), meta, decs, batch_size=3, perm_seed=7)
    for x, y in zip(a, b):
        assert abs(x - y).max() < 1e-4                                                # position bias is absent by design


def test_training_step_updates_head_and_backbone_with_sampled_softmax():
    m = tiny_model().train()
    f = fmt(300)
    c = Collator(f, True, max_train_options=3)
    decs = [(mk(id=f"x{i}", state=f"state {i}", options=list("abcdef"), target=[0, 0, 1, 0, 0, 0]), i) for i in range(4)]
    batch, ds, _ = c(decs)
    before = m.head[1].weight.clone(), next(m.lm.model.L_module.parameters()).clone()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    logits = m(batch).float() + batch["logit_offset"]
    loss, _, _ = decision_loss(logits, batch)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss) and not torch.equal(before[0], m.head[1].weight)
    assert not torch.equal(before[1], next(m.lm.model.L_module.parameters()))
