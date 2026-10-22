"""Mimir basics: PrefixLM attention (bidirectional prefix, causal elsewhere, padding), tokenizer and chat template."""
import pytest
import torch
from transformers.models.hrm_text.configuration_hrm_text import HrmTextConfig
from transformers.models.hrm_text.modeling_hrm_text import HrmTextModel

REV = "521b40b36a79918014544b970d4c2669ff1530eb"


def tiny_backbone():
    torch.manual_seed(0)
    cfg = HrmTextConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                        num_key_value_heads=2, head_dim=16, H_cycles=2, L_cycles=3, L_bp_cycles=[3, 3],
                        max_position_embeddings=64, prefix_lm=True, pad_token_id=0, bos_token_id=2, eos_token_id=1,
                        initializer_range=0.2)
    return HrmTextModel(cfg).eval()


def hidden(m, ids, types):
    ids = torch.tensor([ids])
    with torch.no_grad():
        return m(input_ids=ids, attention_mask=torch.ones_like(ids),
                 token_type_ids=torch.tensor([types])).last_hidden_state[0]


def test_prefix_block_is_bidirectional_and_the_rest_is_causal():
    m = tiny_backbone()
    a, b = [2, 10, 11, 12, 13, 14], [2, 10, 11, 12, 13, 33]            # only the LAST token differs
    # all tokens in the prefix block (token_type_ids = 1, what our formatters use): earlier states see the last token
    h1, h2 = hidden(m, a, [1] * 6), hidden(m, b, [1] * 6)
    assert not torch.allclose(h1[:5], h2[:5], atol=1e-5)
    # no prefix (token_type_ids = 0): causal, earlier states must not change
    c1, c2 = hidden(m, a, [0] * 6), hidden(m, b, [0] * 6)
    assert torch.allclose(c1[:5], c2[:5], atol=1e-6) and not torch.allclose(c1[5], c2[5], atol=1e-5)
    # prefix of 4 tokens, causal response: the prefix sees the response never, the response sees the prefix
    p1, p2 = hidden(m, a, [1, 1, 1, 1, 0, 0]), hidden(m, b, [1, 1, 1, 1, 0, 0])
    assert torch.allclose(p1[:5], p2[:5], atol=1e-6)


def test_right_padding_with_the_attention_mask_does_not_change_real_tokens():
    m = tiny_backbone()
    ids = [2, 10, 11, 12, 13]
    alone = hidden(m, ids, [1] * 5)
    padded_ids = torch.tensor([ids + [0, 0, 0]])
    att = torch.tensor([[1] * 5 + [0] * 3])
    with torch.no_grad():
        out = m(input_ids=padded_ids, attention_mask=att, token_type_ids=att.clone()).last_hidden_state[0, :5]
    assert torch.allclose(alone, out, atol=1e-5)


@pytest.mark.network
def test_tokenizer_and_chat_template_of_the_pinned_revision():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("danish-foundation-models/DFM-Mimir-v1.5", revision=REV)
    assert len(tok) == 262144 and tok.bos_token_id == 2 and tok.pad_token_id == 0 and tok.eos_token == "<turn|>"
    assert tok("hi")["input_ids"][0] == 2                                   # the tokenizer adds <bos> by default ...
    assert tok("hi", add_special_tokens=False)["input_ids"][0] != 2         # ... our formatters add it once themselves
    assert tok.convert_tokens_to_ids("<unused0>") == 6
    msgs = [{"role": "user", "content": "Q?"}]
    plain = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    assert plain == tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert plain == "<bos><|turn>user\nQ?<turn|>\n<|turn>model\n"          # prefix of HRM-Text training = this string
    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": "A"}], tokenize=False)
    assert full.startswith(plain)                                           # the response follows the generation prompt
