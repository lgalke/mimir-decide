"""Prompt rendering + batching for the slot model and the letter baseline.

Slot format (one decision = one sequence, so questions stay independent):

    <bos>State:\n{state}\n\nQuestion ({kind}):\n{question}\n\nOptions:\n[M] {opt1}\n[M] {opt2}\n...

`[M]` is a placeholder token whose embedding is REPLACED by a learned marker vector in the model (no vocabulary
resize needed); the readout is the final hidden state at each marker position. The whole sequence is one
bidirectional PrefixLM block (token_type_ids = 1 on every real token).

Letter-baseline format (Gemma chat template, verified identical to tokenizer.apply_chat_template):

    <bos><|turn>user\nState:..\n\nQuestion:..\n\nOptions:\nA. ..\nB. ..\n\nAnswer with the letter of the correct option.<turn|>\n<|turn>model\n

and the prediction is the logits of the letter tokens at the last position.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import torch

from .schema import Decision

KIND_ID = {"noul": 0, "choice": 1, "score": 2}
LETTERS = [chr(ord("A") + i) for i in range(26)]
MAX_OPTION_TOKENS = 160
MIDDLE = " […] "


@dataclass
class Encoded:
    input_ids: list[int]
    slot_pos: list[int]        # marker positions (slot model) in presented order
    perm: list[int]            # presented option j is original option perm[j]
    n_options: int


class Formatter:
    def __init__(self, tok, mode: str, max_len: int, marker_id: int):
        assert mode in ("slot", "letter")
        self.tok, self.mode, self.max_len, self.marker_id = tok, mode, max_len, marker_id
        self.bos = tok.bos_token_id if tok.bos_token_id is not None else 2
        self._cache: dict = {}

    def _ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def _truncate_state(self, ids: list[int], budget: int) -> list[int]:
        if len(ids) <= budget:
            return ids
        mid = self._ids(MIDDLE)
        keep = max(budget - len(mid), 2)
        head = keep // 2
        return ids[:head] + mid + ids[len(ids) - (keep - head):]

    def encode(self, d: Decision, perm: Optional[list[int]] = None) -> Optional[Encoded]:
        k = len(d.options)
        perm = list(range(k)) if perm is None else perm
        opts = [d.options[i] for i in perm]
        end: list[int] = []
        if self.mode == "slot":
            head = [self.bos] + self._ids("State:\n")
            mid = self._ids(f"\n\nQuestion ({d.kind}):\n{d.question.strip()}\n\nOptions:\n")
            opt_ids = []
            for o in opts:
                body = self._ids(f" {o}\n")[:MAX_OPTION_TOKENS]
                opt_ids.append([self.marker_id] + body)
            tail_len = len(mid) + sum(len(x) for x in opt_ids)
        else:
            head = [self.bos] + self._ids("<|turn>user\nState:\n")
            q_text = f"\n\nQuestion:\n{d.question.strip()}\n\nOptions:\n"
            mid = self._ids(q_text)
            # Segment boundaries are chosen so that the concatenation equals the tokenizer's encoding of the full
            # text (e.g. "\n\n" is ONE token): the last option carries no trailing newline, `end` starts with "\n\n".
            opt_ids = [self._ids(f"{LETTERS[j]}. {o}" + ("\n" if j < k - 1 else ""))[:MAX_OPTION_TOKENS]
                       for j, o in enumerate(opts)]
            end = self._ids("\n\nAnswer with the letter of the correct option.<turn|>\n<|turn>model\n")
            tail_len = len(mid) + sum(len(x) for x in opt_ids) + len(end)
        budget = self.max_len - len(head) - tail_len
        if budget < 16:  # question + options alone do not leave room for any state: drop (counted by caller)
            return None
        state_ids = self._truncate_state(self._ids(d.state.strip()), budget)
        ids = head + state_ids + mid
        slot_pos = []
        for oi in opt_ids:
            if self.mode == "slot":
                slot_pos.append(len(ids))
            ids += oi
        if self.mode == "letter":
            ids += end
        return Encoded(ids, slot_pos, perm, k)


def make_perm(d: Decision, rng: random.Random, augment: bool) -> list[int]:
    k = len(d.options)
    perm = list(range(k))
    if not augment:
        return perm
    if d.kind == "score":
        return perm[::-1] if rng.random() < 0.5 else perm  # ordering semantics preserved up to reversal
    rng.shuffle(perm)
    return perm


class Collator:
    """Turns (Decision, seed) pairs into padded tensors. Used as DataLoader collate_fn."""

    def __init__(self, formatter: Formatter, augment: bool, pad_id: int = 0):
        self.f, self.augment, self.pad_id = formatter, augment, pad_id

    def __call__(self, items):
        encs, decs, drop = [], [], 0
        for it in items:
            d, seed = it if isinstance(it, tuple) else (it, 0)
            rng = random.Random(seed)
            e = self.f.encode(d, make_perm(d, rng, self.augment))
            if e is None:
                drop += 1
                continue
            encs.append(e)
            decs.append(d)
        if not encs:  # every item in this batch was too long: caller skips it
            return None, [], drop
        return self.pack(encs, decs), decs, drop

    def pack(self, encs: list[Encoded], decs: list[Decision]) -> dict:
        B = len(encs)
        T = max(len(e.input_ids) for e in encs)
        K = max(e.n_options for e in encs)
        ids = torch.full((B, T), self.pad_id, dtype=torch.long)
        att = torch.zeros((B, T), dtype=torch.long)
        slot_pos = torch.zeros((B, K), dtype=torch.long)
        valid = torch.zeros((B, K), dtype=torch.bool)
        target = torch.zeros((B, K), dtype=torch.float32)
        last = torch.zeros(B, dtype=torch.long)
        kind = torch.zeros(B, dtype=torch.long)
        for b, (e, d) in enumerate(zip(encs, decs)):
            n = len(e.input_ids)
            ids[b, :n] = torch.tensor(e.input_ids)
            att[b, :n] = 1
            last[b] = n - 1
            valid[b, : e.n_options] = True
            target[b, : e.n_options] = torch.tensor([d.target[i] for i in e.perm], dtype=torch.float32)
            if e.slot_pos:
                slot_pos[b, : e.n_options] = torch.tensor(e.slot_pos)
            kind[b] = KIND_ID[d.kind]
        return {
            "input_ids": ids, "attention_mask": att, "token_type_ids": att.clone(), "slot_pos": slot_pos,
            "valid": valid, "target": target, "last_pos": last, "kind": kind,
            "perm": [e.perm for e in encs],
        }
