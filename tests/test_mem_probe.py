import torch
from transformers.models.hrm_text.configuration_hrm_text import HrmTextConfig
from transformers.models.hrm_text.modeling_hrm_text import HrmTextForCausalLM

from conftest import FakeTok
from mimir_decide import mem_probe
from mimir_decide.formatting import Collator, Formatter
from mimir_decide.model import SlotDecisionModel
from mimir_decide.train import backbone_groups


def test_sweep_stops_at_the_first_out_of_memory_and_reports_speed():
    def run(B):
        if B > 8:
            raise torch.OutOfMemoryError("fake")
        return 0.5 * B
    rows = mem_probe.sweep(run, [1, 4, 8, 12, 16], "cpu")
    assert [r["batch_size"] for r in rows] == [1, 4, 8, 12] and rows[-1] == {"batch_size": 12, "ok": False}
    assert rows[1]["ok"] and rows[1]["ex_per_s"] == 2.0


def test_probe_decision_fills_the_context_exactly_and_the_step_runs_on_a_tiny_model():
    tok = FakeTok()
    max_len = 160
    d = mem_probe.full_length_decision(3)
    f = Formatter(tok, "slot", max_len, 5)
    enc = f.encode(d)
    assert len(enc.input_ids) == max_len                 # the long state is cut to exactly max_len tokens
    cfg = HrmTextConfig(vocab_size=9000, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                        num_key_value_heads=2, head_dim=16, H_cycles=2, L_cycles=3, L_bp_cycles=[3, 3],
                        max_position_embeddings=256, prefix_lm=True, pad_token_id=0, bos_token_id=2, eos_token_id=1,
                        initializer_range=0.02)
    model = SlotDecisionModel(HrmTextForCausalLM(cfg))
    model.train()
    groups, _ = backbone_groups(model.lm, {"lr_backbone": 1e-4})
    groups.append({"params": model.new_parameters(), "lr": 1e-3, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    step = mem_probe.make_step_fn(model, opt, groups, Collator(f, False), d, "cpu")
    rows = mem_probe.sweep(step, [1, 2], "cpu")
    assert all(r["ok"] for r in rows) and rows[0]["step_s"] > 0
