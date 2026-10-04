import copy
import random

import numpy as np
import pytest
import torch
from transformers.models.hrm_text.configuration_hrm_text import HrmTextConfig
from transformers.models.hrm_text.modeling_hrm_text import HrmTextForCausalLM

from mimir_decide import metrics
from mimir_decide.formatting import Collator, Formatter
from mimir_decide.model import LetterBaseline, SlotDecisionModel, decision_loss

MARKER = 5


def tiny_lm(seed=0):
    torch.manual_seed(seed)
    cfg = HrmTextConfig(vocab_size=200, hidden_size=48, intermediate_size=96, num_hidden_layers=1,
                        num_attention_heads=4, num_key_value_heads=4, head_dim=12, H_cycles=2, L_cycles=3,
                        L_bp_cycles=[3, 3], max_position_embeddings=512, prefix_lm=True, pad_token_id=0,
                        bos_token_id=2, eos_token_id=1, initializer_range=0.0255)
    return HrmTextForCausalLM(cfg)


def synth(mk, n=24):
    """Label is determined by a keyword in the state; options shuffled so position carries no signal."""
    rng = random.Random(0)
    out = []
    for i in range(n):
        gold = rng.randrange(3)
        out.append(mk(id=f"s{i}", state=f"key {'abc'[gold]} item {i}", question="which letter?",
                      options=["a", "b", "c"], target=[float(j == gold) for j in range(3)]))
    return out


def _train(model, decs, fmt, steps, lr, augment=True):
    col = Collator(fmt, augment=augment)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    losses = []
    for s in range(steps):
        batch, kept, _ = col([(d, s * 100 + i) for i, d in enumerate(decs)])
        loss, _, _ = decision_loss(model(batch).float(), batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def test_slot_model_learns_and_backbone_gets_gradients(tok, mk):
    decs = synth(mk)
    model = SlotDecisionModel(tiny_lm())
    fmt = Formatter(tok, "slot", 128, MARKER)
    before = copy.deepcopy({k: v.clone() for k, v in model.lm.model.L_module.state_dict().items()})
    emb_before = model.lm.model.embed_tokens.weight.clone()
    losses = _train(model, decs, fmt, steps=120, lr=3e-3)
    assert losses[0] > 0.9 and losses[-1] < 0.35 * losses[0], (losses[0], losses[-1])
    after = model.lm.model.L_module.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)        # backbone trained
    assert torch.equal(emb_before, model.lm.model.embed_tokens.weight)      # frozen embeddings untouched
    assert not model.lm.lm_head.weight.requires_grad


def test_letter_baseline_learns(tok, mk):
    decs = synth(mk)
    model = LetterBaseline(tok, tiny_lm(1))
    fmt = Formatter(tok, "letter", 160, MARKER)
    # fixed option order: with shuffled options a 1-layer toy model must also learn keyword->letter binding,
    # which is far slower (loss 1.12 -> 0.81 after 400 steps in a manual run); that gap is what the slot head avoids.
    losses = _train(model, decs, fmt, steps=150, lr=3e-3, augment=False)
    assert losses[-1] < 0.5 * losses[0], (losses[0], losses[-1])


def test_slot_logits_ignore_padding_and_batch_composition(tok, mk):
    """A decision's logits must not depend on what else is in the batch (independence + right-padding)."""
    decs = synth(mk, 4)
    model = SlotDecisionModel(tiny_lm()).eval()
    f = Formatter(tok, "slot", 128, MARKER)
    col = Collator(f, False)
    with torch.no_grad():
        solo = model(col.pack([f.encode(decs[0])], [decs[0]]))[0, :3]
        long_d = mk(id="L", state="x" * 60, options=["a", "b", "c"], target=[1, 0, 0])
        both = model(col.pack([f.encode(decs[0]), f.encode(long_d)], [decs[0], long_d]))[0, :3]
    assert torch.allclose(solo, both, atol=1e-4)


def test_decision_loss_soft_target_and_emd(mk):
    logits = torch.log(torch.tensor([[0.2, 0.8, 1e-9]]))
    batch = {"target": torch.tensor([[0.2, 0.8, 0.0]]), "kind": torch.tensor([1])}
    loss, ce, emd = decision_loss(logits, batch)
    assert ce.item() == pytest.approx(-(0.2 * np.log(0.2) + 0.8 * np.log(0.8)), abs=1e-4) and emd.item() < 1e-6
    # EMD only applied to score rows; invariant to reversing the presentation order
    batch_s = {"target": torch.tensor([[0.7, 0.2, 0.1]]), "kind": torch.tensor([2])}
    lg = torch.tensor([[0.0, 1.0, 2.0]])
    l1, _, e1 = decision_loss(lg, batch_s)
    l2, _, e2 = decision_loss(lg.flip(-1), {**batch_s, "target": batch_s["target"].flip(-1)})
    assert e1.item() == pytest.approx(e2.item()) and l1.item() > decision_loss(lg, {**batch_s, "kind": torch.tensor([1])})[0].item()


def test_fit_temperature_recovers_known_temperature():
    rng = np.random.default_rng(0)
    K, N, T_true = 4, 4000, 2.5
    z = rng.normal(size=(N, K)) * 3                      # model logits that are overconfident by factor T_true
    p_true = np.exp(z / T_true); p_true /= p_true.sum(1, keepdims=True)
    y = np.array([rng.choice(K, p=p) for p in p_true])
    targets = [np.eye(K)[i] for i in y]
    T = metrics.fit_temperature(list(z), targets)
    assert T == pytest.approx(T_true, rel=0.12)
    # calibration improves
    from mimir_decide.schema import Decision
    decs = [Decision("i", "c", "g", "d", "d/s", "validation", "en", "mit", "choice", "s", "q", list("abcd"),
                     t.tolist()) for t in targets]
    e0 = metrics.evaluate(list(z), decs)["all"]["ece"]
    e1 = metrics.evaluate(list(z), decs, {"choice": T})["all"]["ece"]
    assert e1 < e0 and e1 < 0.03


def test_metric_basics():
    assert metrics.top1_correct(np.array([0.5, 0.5]), np.array([0.5, 0.5]))            # ties both count
    assert not metrics.top1_correct(np.array([0.9, 0.1]), np.array([0.0, 1.0]))
    conf, corr = np.array([0.9] * 10), np.array([1.0] * 9 + [0.0])
    assert metrics.ece(conf, corr) == pytest.approx(0.0, abs=1e-9)
    assert metrics.spearman(np.arange(10.0), np.arange(10.0) * 2) == pytest.approx(1.0)
    sr = metrics.selective_risk(np.array([0.9, 0.8, 0.1, 0.2]), np.array([1.0, 1.0, 0.0, 0.0]))
    assert sr["risk@0.5"] == 0.0 and sr["risk@1.0"] == 0.5
