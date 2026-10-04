import pytest
import torch

from conftest import make_decision as mk


@pytest.mark.network  # needs the real tokenizer (for save_pretrained); tiny random model, CPU
@pytest.mark.parametrize("model_type", ["slot", "letter"])
def test_save_load_roundtrip_gives_same_logits(tmp_path, model_type):
    from transformers import AutoTokenizer
    from transformers.models.hrm_text.configuration_hrm_text import HrmTextConfig
    from transformers.models.hrm_text.modeling_hrm_text import HrmTextForCausalLM

    from mimir_decide.formatting import Collator, Formatter
    from mimir_decide.model import LetterBaseline, SlotDecisionModel, load_model, save_model

    tok = AutoTokenizer.from_pretrained("danish-foundation-models/DFM-Mimir-v1.5",
                                        revision="521b40b36a79918014544b970d4c2669ff1530eb")
    torch.manual_seed(0)
    cfg = HrmTextConfig(vocab_size=262144, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                        num_attention_heads=4, num_key_value_heads=4, head_dim=16, H_cycles=2, L_cycles=3,
                        L_bp_cycles=[3, 3], max_position_embeddings=512, prefix_lm=True, pad_token_id=0,
                        bos_token_id=2, eos_token_id=106, initializer_range=0.0255)
    lm = HrmTextForCausalLM(cfg)
    model = SlotDecisionModel(lm) if model_type == "slot" else LetterBaseline(tok, lm)
    if model_type == "slot":  # make the new parameters non-trivial so the test would catch a lost head/marker
        with torch.no_grad():
            model.marker.add_(0.3)
            model.head[1].weight.normal_(0, 0.5)
            model.log_scale.add_(torch.tensor([0.1, -0.2, 0.3]))
    marker_id = tok.convert_tokens_to_ids("<unused0>")
    meta = {"model_type": model_type, "max_len": 256, "marker_id": marker_id}
    save_model(model, tok, tmp_path / "ck", meta)

    d = mk(state="Hello world, this is a state.", question="Pick one?", options=["red", "green", "blue"],
           target=[0, 1, 0])
    f = Formatter(tok, "slot" if model_type == "slot" else "letter", 256, marker_id)
    batch = Collator(f, False).pack([f.encode(d)], [d])
    model.eval()
    with torch.no_grad():
        ref = model(batch)[0, :3]
    loaded, tok2, meta2 = load_model(tmp_path / "ck", "cpu", dtype=torch.float32)
    assert meta2["model_type"] == model_type and meta2["marker_id"] == marker_id
    with torch.no_grad():
        got = loaded(batch)[0, :3]
    # weights are stored in bf16, so allow bf16 rounding
    assert torch.allclose(ref, got, atol=0.05, rtol=0.05), (ref, got)
    assert (torch.softmax(ref, -1).argmax() == torch.softmax(got, -1).argmax())
