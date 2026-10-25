"""Span-pooling readout with per-kind heads (D35)."""
import torch
from transformers.models.hrm_text.configuration_hrm_text import HrmTextConfig
from transformers.models.hrm_text.modeling_hrm_text import HrmTextForCausalLM

from conftest import FakeTok, make_decision as mk
from mimir_decide.formatting import CHAT_END, CHAT_HEAD, Collator, Formatter, mode_for
from mimir_decide.model import SpanPoolModel, decision_loss


def tiny_lm(seed=0):
    torch.manual_seed(seed)
    cfg = HrmTextConfig(vocab_size=9000, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                        num_key_value_heads=2, head_dim=16, H_cycles=2, L_cycles=3, L_bp_cycles=[3, 3],
                        max_position_embeddings=512, prefix_lm=True, pad_token_id=0, bos_token_id=2, eos_token_id=1,
                        initializer_range=0.02)
    return HrmTextForCausalLM(cfg)


def ids(s):
    return FakeTok()(s)["input_ids"]


def test_mode_and_spans_cover_exactly_the_option_text():
    assert mode_for("span") == "span"
    f = Formatter(FakeTok(), "span", 400, 5, chat=True)
    d = mk(state="s", options=["red", "blue", "green"], target=[1, 0, 0])
    e = f.encode(d)
    assert [e.input_ids[a:b] for a, b in e.spans] == [ids("red"), ids("blue"), ids("green")]
    assert e.slot_pos == [] and 5 not in e.input_ids                              # no marker tokens
    assert e.input_ids[: 1 + len(ids(CHAT_HEAD))] == [2] + ids(CHAT_HEAD) and e.input_ids[-len(ids(CHAT_END)):] == ids(CHAT_END)
    assert e.input_ids[e.spans[0][1]] == ids("\n")[0] and e.input_ids[e.spans[2][1]:][: len(ids(CHAT_END))] == ids(CHAT_END)  # last option: no trailing newline
    plain = Formatter(FakeTok(), "span", 400, 5).encode(d)
    assert plain.input_ids[-1] == ids("\n")[0] and plain.input_ids != e.input_ids


def test_permutation_moves_the_spans_with_the_options():
    f = Formatter(FakeTok(), "span", 400, 5, chat=True)
    d = mk(options=["red", "blue", "green"], target=[1, 0, 0])
    e = f.encode(d, [2, 0, 1])
    assert [e.input_ids[a:b] for a, b in e.spans] == [ids("green"), ids("red"), ids("blue")]


def test_pack_gives_span_tensors_and_the_pooled_vector_is_the_mean_over_the_text():
    f = Formatter(FakeTok(), "span", 400, 5, chat=True)
    d = mk(state="state", options=["red", "blue"], target=[0, 1])
    c = Collator(f, False)
    e = f.encode(d)
    batch = c.pack([e], [d])
    assert batch["span_start"].tolist() == [[e.spans[0][0], e.spans[1][0]]] and batch["span_end"][0, 1] == e.spans[1][1]
    lm = tiny_lm()
    m = SpanPoolModel(lm)
    m.eval()
    with torch.no_grad():
        h = lm.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                     token_type_ids=batch["token_type_ids"]).last_hidden_state[0].float()
        manual = h[e.spans[1][0]:e.spans[1][1]].mean(0)
        # one head per kind: make the choice head (index 1) a pure projection and compare with the manual mean
        for hd in m.heads:
            torch.nn.init.zeros_(hd[1].weight); hd[1].bias.zero_()
        m.heads[1][0].weight.fill_(1.0); m.heads[1][0].bias.zero_()
        m.heads[1][1].weight[0, 0] = 1.0
        out = m(batch)[0, 1]
        ln = torch.nn.functional.layer_norm(manual, (manual.numel(),), weight=torch.ones(32), bias=torch.zeros(32))
    assert torch.allclose(out, ln[0], atol=1e-4)


def test_each_kind_uses_its_own_head_and_a_shared_head_uses_the_scale():
    f = Formatter(FakeTok(), "span", 400, 5, chat=True)
    ds = [mk(id=k, kind=k, options=["a", "b"] if k != "score" else list("012"), target=[1, 0] if k != "score" else [0, 1, 0],
             option_values=None if k != "score" else [0., 1., 2.]) for k in ("noul", "choice", "score")]
    batch = Collator(f, False).pack([f.encode(d) for d in ds], ds)
    m = SpanPoolModel(tiny_lm()).eval()
    with torch.no_grad():
        base = m(batch).clone()
        m.heads[2][1].bias.add_(5.0)                                  # only the score head changes
        moved = m(batch)
    assert torch.allclose(moved[0], base[0]) and torch.allclose(moved[1], base[1])      # noul and choice unchanged
    assert torch.allclose(moved[2, :3] - base[2, :3], torch.full((3,), 5.0), atol=1e-4)  # the score head shifted its scores
    shared = SpanPoolModel(tiny_lm(), per_kind_heads=False).eval()
    assert len(shared.heads) == 1 and any(p is shared.log_scale for p in shared.new_parameters())
    assert shared(batch).shape == base.shape and len(SpanPoolModel(tiny_lm()).heads) == 3


def test_frozen_backbone_probe_trains_only_the_heads():
    f = Formatter(FakeTok(), "span", 400, 5, chat=True)
    ds = [mk(id=f"d{i}", state=f"state {i}", options=["a", "b", "c"], target=[1, 0, 0]) for i in range(4)]
    batch = Collator(f, False).pack([f.encode(d) for d in ds], ds)
    m = SpanPoolModel(tiny_lm(), freeze_backbone=True)
    assert not any(p.requires_grad for p in m.lm.parameters())
    params = m.new_parameters()
    assert params and all(p.requires_grad for p in params)
    loss, _, _ = decision_loss(m(batch).float(), batch)
    loss.backward()
    assert m.heads[1][1].weight.grad is not None and all(p.grad is None for p in m.lm.parameters())


def test_full_training_step_updates_backbone_and_heads():
    f = Formatter(FakeTok(), "span", 400, 5, chat=True)
    ds = [mk(id=f"d{i}", state=f"state {i}", options=["a", "b", "c"], target=[0, 1, 0]) for i in range(4)]
    batch = Collator(f, True).pack([f.encode(d) for d in ds], ds)
    m = SpanPoolModel(tiny_lm()).train()
    before = next(m.lm.model.L_module.parameters()).clone()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    loss, _, _ = decision_loss(m(batch).float(), batch)
    loss.backward(); opt.step()
    assert torch.isfinite(loss) and not torch.equal(before, next(m.lm.model.L_module.parameters()))
