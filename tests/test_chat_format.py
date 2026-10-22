"""Chat format of the slot and scalar prompts (D34): the Gemma user turn plus generation prompt, as in Mimir's training."""
import pytest
import torch

from conftest import FakeTok, make_decision as mk
from mimir_decide.formatting import CHAT_END, CHAT_HEAD, Collator, Formatter
from test_scalar_head import tiny_model


def ids(s):
    return FakeTok()(s)["input_ids"]


def test_scalar_rows_are_wrapped_in_the_user_turn_and_end_with_the_generation_prompt():
    f = Formatter(FakeTok(), "scalar", 400, 5, chat=True)
    d = mk(state="the state", options=["red", "blue"], target=[1, 0])
    e = f.encode(d)
    assert f.chat
    for r, o in zip(e.rows, ["red", "blue"]):
        assert r[: 1 + len(ids(CHAT_HEAD))] == [2] + ids(CHAT_HEAD)
        assert r[-len(ids(CHAT_END)):] == ids(CHAT_END)
        assert r[-len(ids(CHAT_END)) - len(ids(o)): -len(ids(CHAT_END))] == ids(o)      # option just before <turn|>
    plain = Formatter(FakeTok(), "scalar", 400, 5).encode(d)
    assert plain.rows[0][-3:] == ids("red")[-3:] and plain.rows[0] != e.rows[0]


def test_slot_chat_keeps_markers_adds_the_generation_prompt_and_trims_the_last_option():
    f = Formatter(FakeTok(), "slot", 400, 5, chat=True)
    d = mk(state="s", options=["red", "blue", "green"], target=[1, 0, 0])
    e = f.encode(d)
    assert e.input_ids[: 1 + len(ids(CHAT_HEAD))] == [2] + ids(CHAT_HEAD)
    assert e.input_ids[-len(ids(CHAT_END)):] == ids(CHAT_END)
    assert [e.input_ids[p] for p in e.slot_pos] == [5, 5, 5]
    assert e.input_ids[-len(ids(CHAT_END)) - 1] == ids("n")[0]                            # "green" ends the user content
    plain = Formatter(FakeTok(), "slot", 400, 5).encode(d)
    assert e.input_ids != plain.input_ids and plain.input_ids[-1] == ids("\n")[0]


def test_letter_is_always_chat_and_plain_default_is_unchanged():
    assert Formatter(FakeTok(), "letter", 100, 5).chat is True
    assert Formatter(FakeTok(), "slot", 100, 5).chat is False and Formatter(FakeTok(), "scalar", 100, 5).chat is False


def test_scalar_chat_scores_do_not_depend_on_option_order_or_padding():
    m = tiny_model()
    f = Formatter(FakeTok(), "scalar", 300, 5, chat=True)
    c = Collator(f, False)
    d = mk(state="a state about colours", options=["red", "green", "blue"], target=[0, 1, 0])
    other = mk(id="o", state="word " * 25, options=["a", "b"], target=[1, 0])
    with torch.no_grad():
        base = m(c.pack([f.encode(d)], [d]))[0, :3]
        rev = m(c.pack([f.encode(d, [2, 1, 0])], [d]))[0, :3]
        padded = m(c.pack([f.encode(other), f.encode(d)], [other, d]))[1, :3]
    assert torch.allclose(base, rev.flip(0), atol=1e-4) and torch.allclose(base, padded, atol=1e-4)


@pytest.mark.network
def test_real_tokenizer_chat_variants_equal_the_template_rendering():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("danish-foundation-models/DFM-Mimir-v1.5",
                                        revision="521b40b36a79918014544b970d4c2669ff1530eb")
    marker = tok.convert_tokens_to_ids("<unused0>")

    def ref(content):
        r = tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True)
        return r if isinstance(r, list) else r["input_ids"]

    d = mk(state="Hello world", question="Which colour?", options=["red", "green", "blue"], target=[0, 1, 0])
    row = Formatter(tok, "scalar", 512, marker, chat=True).encode(d).rows[1]
    assert row == ref("State:\nHello world\n\nQuestion (choice):\nWhich colour?\n\nOption:\ngreen")
    e = Formatter(tok, "slot", 512, marker, chat=True).encode(d)
    # the marker token has no text form; without the markers the ids equal the template rendering of the options text
    expected = ref("State:\nHello world\n\nQuestion (choice):\nWhich colour?\n\nOptions:\n red\n green\n blue")
    assert [i for i in e.input_ids if i != marker] == expected and [e.input_ids[p] for p in e.slot_pos] == [marker] * 3
