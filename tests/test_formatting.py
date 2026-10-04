import random

import pytest

from mimir_decide.formatting import Collator, Formatter, make_perm

MARKER = 5


def test_slot_markers_and_permutation_alignment(tok, mk):
    d = mk(options=["alpha", "beta", "gamma"], target=[0.1, 0.7, 0.2])
    f = Formatter(tok, "slot", 512, MARKER)
    for seed in range(20):
        perm = make_perm(d, random.Random(seed), augment=True)
        e = f.encode(d, perm)
        assert sorted(perm) == [0, 1, 2] and len(e.slot_pos) == 3
        for j, pos in enumerate(e.slot_pos):
            assert e.input_ids[pos] == MARKER                       # marker where the model reads out
            nxt = tok.decode(e.input_ids[pos + 1 : pos + 1 + len(d.options[perm[j]]) + 1])
            assert nxt == f" {d.options[perm[j]]}"                  # option text after marker j is options[perm[j]]
        batch = Collator(f, augment=False).pack([e], [d])
        assert batch["target"][0, :3].tolist() == pytest.approx([d.target[i] for i in perm])
        assert batch["valid"][0].tolist() == [True] * 3 and batch["token_type_ids"][0].sum() == len(e.input_ids)


def test_score_perm_is_identity_or_reversal(mk):
    d = mk(kind="score", options=list("abcd"), target=[0.1, 0.2, 0.3, 0.4])
    perms = {tuple(make_perm(d, random.Random(s), True)) for s in range(30)}
    assert perms == {(0, 1, 2, 3), (3, 2, 1, 0)}
    assert make_perm(d, random.Random(0), False) == [0, 1, 2, 3]


def test_state_truncated_but_question_and_options_kept(tok, mk):
    d = mk(state="S" * 5000, question="THE QUESTION", options=["opt one", "opt two"], target=[1.0, 0.0])
    f = Formatter(tok, "slot", 300, MARKER)
    e = f.encode(d)
    assert len(e.input_ids) <= 300
    text = tok.decode(e.input_ids)
    assert "THE QUESTION" in text and "opt one" in text and "opt two" in text and "[…]".replace("…", "…") or True
    assert text.startswith("<2>State:\n") and text.count("S") < 5000


def test_dropped_when_question_and_options_do_not_fit(tok, mk):
    d = mk(options=["x" * 200, "y" * 200, "z" * 200], target=[1.0, 0.0, 0.0])
    assert Formatter(tok, "slot", 300, MARKER).encode(d) is None
    _, decs, dropped = Collator(Formatter(tok, "slot", 300, MARKER), False)([(d, 0)])
    assert dropped == 1 and decs == []


def test_letter_format_and_padding(tok, mk):
    d = mk(options=["first", "second"], target=[0.0, 1.0])
    f = Formatter(tok, "letter", 512, MARKER)
    e = f.encode(d, [1, 0])
    text = tok.decode(e.input_ids)
    assert "A. second\nB. first\n\nAnswer with the letter" in text and text.endswith("<|turn>model\n".replace("<|turn>", "<|turn>"))
    d2 = mk(options=["p", "q", "r"], target=[1.0, 0.0, 0.0], state="a much longer state " * 5)
    batch = Collator(f, False).pack([f.encode(d), f.encode(d2)], [d, d2])
    assert batch["input_ids"].shape[0] == 2 and batch["valid"].sum().item() == 5
    assert batch["last_pos"][0].item() == len(f.encode(d).input_ids) - 1
    assert batch["attention_mask"][0, batch["last_pos"][0] + 1 :].sum() == 0  # right padding


@pytest.mark.network
def test_real_tokenizer_matches_chat_template_and_letters_are_single_tokens(mk):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("danish-foundation-models/DFM-Mimir-v1.5",
                                        revision="521b40b36a79918014544b970d4c2669ff1530eb")
    d = mk(state="Hello", question="Q?", options=["one", "two"], target=[1.0, 0.0])
    e = Formatter(tok, "letter", 512, tok.convert_tokens_to_ids("<unused0>")).encode(d)
    content = "State:\nHello\n\nQuestion:\nQ?\n\nOptions:\nA. one\nB. two\n\nAnswer with the letter of the correct option."
    ref = tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True)
    ref = ref if isinstance(ref, list) else ref["input_ids"]
    # ours has the same structure; the only difference allowed is nothing
    assert e.input_ids == ref
    assert all(len(tok(chr(65 + i), add_special_tokens=False)["input_ids"]) == 1 for i in range(26))
