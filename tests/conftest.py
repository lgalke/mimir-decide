import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeTok:
    """Reversible ASCII tokenizer: id = ord(c) + 10 (ids 10..137). Specials: pad 0, bos 2, marker 5."""
    bos_token_id, pad_token_id, eos_token_id = 2, 0, 1

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) + 10 for c in text]}

    def decode(self, ids):
        return "".join(chr(i - 10) if i >= 10 else f"<{i}>" for i in ids)


@pytest.fixture
def tok():
    return FakeTok()


def make_decision(**kw):
    from mimir_decide.schema import Decision

    base = dict(id="x:1", case_id="x:1", group_id="g:1", dataset="d", source="d/s", split="train", language="en",
                license="apache-2.0", kind="choice", state="state text", question="which?",
                options=["a", "b", "c"], target=[0.0, 1.0, 0.0], option_values=None, soft=False, variant=None)
    base.update(kw)
    return Decision(**base)


@pytest.fixture
def mk():
    return make_decision
