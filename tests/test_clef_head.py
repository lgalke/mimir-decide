"""Clef-style head (D38): formatter spans, lexical vectors, prior, frozen probe and full training step."""
import torch
import torch.nn.functional as F
from transformers.models.hrm_text.configuration_hrm_text import HrmTextConfig
from transformers.models.hrm_text.modeling_hrm_text import HrmTextForCausalLM

from conftest import FakeTok, make_decision as mk
from mimir_decide.formatting import CHAT_END, CHAT_HEAD, Collator, Formatter, mode_for
from mimir_decide.model import ClefHeadModel, decision_loss

SMALL = {"width": 32, "routing_layers": 1, "layers": 1, "heads": 2, "feedforward": 64}


def tiny_lm(seed=0):
    torch.manual_seed(seed)
    cfg = HrmTextConfig(vocab_size=9000, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                        num_key_value_heads=2, head_dim=16, H_cycles=2, L_cycles=3, L_bp_cycles=[3, 3],
                        max_position_embeddings=512, prefix_lm=True, pad_token_id=0, bos_token_id=2, eos_token_id=1,
                        initializer_range=0.02)
    return HrmTextForCausalLM(cfg)


def ids(s):
    return FakeTok()(s)["input_ids"]


def batch_of(ds, f=None):
    f = f or Formatter(FakeTok(), "clef", 400, 5, chat=True)
    return f, Collator(f, False).pack([f.encode(d) for d in ds], ds)


def test_question_and_option_spans_are_exact():
    assert mode_for("clef") == "clef"
    f = Formatter(FakeTok(), "clef", 400, 5, chat=True)
    d = mk(state="some state", question="Which colour is it?", options=["red", "blue"], target=[1, 0])
    e = f.encode(d)
    assert e.input_ids[e.qspan[0]:e.qspan[1]] == ids("Which colour is it?")
    assert [e.input_ids[a:b] for a, b in e.spans] == [ids("red"), ids("blue")]
    assert e.qspan[1] < e.spans[0][0] and 5 not in e.input_ids
    assert e.input_ids[: 1 + len(ids(CHAT_HEAD))] == [2] + ids(CHAT_HEAD) and e.input_ids[-len(ids(CHAT_END)):] == ids(CHAT_END)
    _, batch = batch_of([d], f)
    assert batch["q_start"].tolist() == [e.qspan[0]] and batch["q_end"].tolist() == [e.qspan[1]] and "span_start" in batch


def test_the_prior_is_the_cosine_between_the_option_embedding_and_the_context():
    d = mk(state="state", question="Which?", options=["red", "blue", "green"], target=[0, 1, 0])
    f, batch = batch_of([d])
    lm = tiny_lm()
    m = ClefHeadModel(lm, clef=SMALL).eval()
    with torch.no_grad():
        m.head.residual_gate.fill_(-60.0)            # sigmoid(gate) = 0: the logits are the lexical prior alone
        m.head.prior_logit_scale.zero_()             # scale exp(0) = 1
        logits = m(batch)[0, :3]
        h = lm.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                     token_type_ids=batch["token_type_ids"]).last_hidden_state.float()
        hn = m.head.hidden_norm(h)[0]
        e = f.encode(d)
        qvec = hn[e.qspan[0]:e.qspan[1]].mean(0)
        anchor = F.normalize(qvec + hn[int(batch["last_pos"][0])], dim=0)
        manual = []
        for a, b in e.spans:
            lex = lm.lm_head.weight[batch["input_ids"][0, a:b]].float().mean(0)
            manual.append(torch.dot(F.normalize(lex, dim=0), anchor))
    assert torch.allclose(logits, torch.stack(manual), atol=1e-4)


def test_logits_are_finite_for_all_kinds_with_different_option_counts_and_invalid_slots_are_masked():
    ds = [mk(id="n", kind="noul", options=["yes", "no"], target=[1, 0]),
          mk(id="c", kind="choice", options=list("abcde"), target=[0, 0, 1, 0, 0]),
          mk(id="s", kind="score", options=list("0123"), target=[0, 1, 0, 0], option_values=[0., 1, 2, 3])]
    _, batch = batch_of(ds)
    out = ClefHeadModel(tiny_lm(), clef=SMALL).eval()(batch)
    assert out.shape == (3, 5) and torch.isfinite(out[batch["valid"]]).all() and (out[~batch["valid"]] < -1e8).all()


def test_frozen_backbone_probe_trains_only_the_head_and_a_full_step_trains_both():
    ds = [mk(id=f"d{i}", state=f"state {i}", options=["a", "b", "c"], target=[1, 0, 0]) for i in range(4)]
    _, batch = batch_of(ds)
    probe = ClefHeadModel(tiny_lm(), freeze_backbone=True, clef=SMALL)
    assert not any(p.requires_grad for p in probe.lm.parameters()) and all(p.requires_grad for p in probe.new_parameters())
    loss, _, _ = decision_loss(probe(batch).float(), batch)
    loss.backward()
    assert all(p.grad is None for p in probe.lm.parameters()) and probe.head.question_projection.weight.grad is not None
    full = ClefHeadModel(tiny_lm(), clef=SMALL).train()
    before = next(full.lm.model.L_module.parameters()).clone()
    opt = torch.optim.AdamW([p for p in full.parameters() if p.requires_grad], lr=1e-2)
    loss, _, _ = decision_loss(full(batch).float(), batch)
    loss.backward(); opt.step()
    assert torch.isfinite(loss) and not torch.equal(before, next(full.lm.model.L_module.parameters()))
    assert full.lm.lm_head.weight.grad is None                                   # the lexical rows stay frozen


def test_padding_does_not_change_the_logits_of_a_decision():
    d = mk(id="a", state="a state", question="Which?", options=["red", "blue"], target=[1, 0])
    long = mk(id="b", state="word " * 30, question="Which one?", options=["x", "y", "z"], target=[1, 0, 0])
    m = ClefHeadModel(tiny_lm(), clef=SMALL).eval()
    f = Formatter(FakeTok(), "clef", 400, 5, chat=True)
    c = Collator(f, False)
    with torch.no_grad():
        alone = m(c.pack([f.encode(d)], [d]))[0, :2]
        padded = m(c.pack([f.encode(long), f.encode(d)], [long, d]))[1, :2]
    assert torch.allclose(alone, padded, atol=1e-4)
