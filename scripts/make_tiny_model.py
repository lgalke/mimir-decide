"""Create a tiny randomly-initialised HRM-Text (real 262144 vocab) for pipeline smoke tests.

    python scripts/make_tiny_model.py ~/mimir-decide-data/smoke/tiny-hrm
"""
import sys
from pathlib import Path

import torch
from transformers.models.hrm_text.configuration_hrm_text import HrmTextConfig
from transformers.models.hrm_text.modeling_hrm_text import HrmTextForCausalLM

out = Path(sys.argv[1]).expanduser()
torch.manual_seed(0)
cfg = HrmTextConfig(
    vocab_size=262144, hidden_size=64, intermediate_size=128, num_hidden_layers=2,  # layers per H/L stack
    num_attention_heads=4, num_key_value_heads=4, head_dim=16, H_cycles=2, L_cycles=3, L_bp_cycles=[3, 3],
    max_position_embeddings=4096, prefix_lm=True, pad_token_id=0, bos_token_id=2, eos_token_id=106,
    initializer_range=0.0255,
)
m = HrmTextForCausalLM(cfg)
m.save_pretrained(out)
print("saved", out, sum(p.numel() for p in m.parameters()) / 1e6, "M params")
