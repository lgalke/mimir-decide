import torch
from transformers.models.hrm_text.configuration_hrm_text import HrmTextConfig
from transformers.models.hrm_text.modeling_hrm_text import HrmTextForCausalLM

from mimir_decide.train import backbone_groups, cycle_dividers


def tiny(bp=(3, 3)):
    cfg = HrmTextConfig(vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                        num_attention_heads=2, num_key_value_heads=2, head_dim=16, H_cycles=2, L_cycles=3,
                        L_bp_cycles=list(bp), max_position_embeddings=64, prefix_lm=True, pad_token_id=0,
                        bos_token_id=2, eos_token_id=1, initializer_range=0.02)
    return HrmTextForCausalLM(cfg)


CFG = {"lr_backbone": 1.0e-5, "weight_decay": 0.01}


def test_flag_off_is_one_group_identical_to_the_earlier_runs():
    lm = tiny()
    groups, div = backbone_groups(lm, CFG)
    assert div is None and len(groups) == 1
    assert groups[0]["lr"] == 1.0e-5 and groups[0]["weight_decay"] == 0.01
    assert [id(p) for p in groups[0]["params"]] == [id(p) for p in lm.parameters() if p.requires_grad]


def test_dividers_for_full_backprop_are_6_and_2():
    lm = tiny((3, 3))
    assert cycle_dividers(lm) == {"L": 6, "H": 2}
    groups, div = backbone_groups(lm, {**CFG, "lr_cycle_dividers": True})
    by = {g["name"]: g for g in groups}
    assert abs(by["L"]["lr"] - 1e-5 / 6) < 1e-15 and abs(by["H"]["lr"] - 1e-5 / 2) < 1e-15
    assert abs(by["other"]["lr"] - 1e-5) < 1e-15           # embeddings / output head are applied once


def test_groups_cover_every_trainable_parameter_once():
    lm = tiny()
    groups, _ = backbone_groups(lm, {**CFG, "lr_cycle_dividers": True})
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == sum(1 for p in lm.parameters() if p.requires_grad)


def test_truncated_backprop_counts_only_gradient_carrying_l_applications():
    assert cycle_dividers(tiny((0, 3))) == {"L": 3, "H": 2}


def test_frozen_backbone_gives_no_groups():
    lm = tiny()
    for p in lm.parameters():
        p.requires_grad = False
    assert backbone_groups(lm, {**CFG, "lr_cycle_dividers": True}) == ([], None)
