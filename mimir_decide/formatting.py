"""Prompt rendering + batching for the slot model and the letter baseline.

Slot format (one decision = one sequence, so questions stay independent):

    <bos>State:\n{state}\n\nQuestion ({kind}):\n{question}\n\nOptions:\n[M] {opt1}\n[M] {opt2}\n...

`[M]` is a placeholder token whose embedding is REPLACED by a learned marker vector in the model (no vocabulary
resize needed); the readout is the final hidden state at each marker position. The whole sequence is one
bidirectional PrefixLM block (token_type_ids = 1 on every real token).

Letter-baseline format (Gemma chat template, verified identical to tokenizer.apply_chat_template):

    <bos><|turn>user\nState:..\n\nQuestion:..\n\nOptions:\nA. ..\nB. ..\n\nAnswer with the letter of the correct option.<turn|>\n<|turn>model\n

and the prediction is the logits of the letter tokens at the last position.

Chat variant of the slot and scalar formats (D34, `chat=True`): the same text inside the Gemma user turn, followed by the
generation prompt, exactly as Mimir saw its training prompts (the prefix of an HRM-Text example is the chat template
rendered with the generation prompt):

    <bos><|turn>user\nState:..\n\nQuestion (kind):..\n\nOptions:\n[M] ..\n[M] ..<turn|>\n<|turn>model\n      (slot)
    <bos><|turn>user\nState:..\n\nQuestion (kind):..\n\nOption:\n{option}<turn|>\n<|turn>model\n             (scalar row)

the slot readout stays at the markers, the scalar readout is the final token of the row (the generation prompt).

Span format (D35): the slot prompt without markers. All options are listed in one sequence (one forward pass); the
readout of an option is the MEAN of the final hidden states over the tokens of its own text (`Encoded.spans`).

    <bos>[<|turn>user\n]State:..\n\nQuestion (kind):..\n\nOptions:\n{opt1}\n{opt2}...[<turn|>\n<|turn>model\n]

Scalar format (D33): one sequence PER OPTION, so a decision with K options becomes K rows that share the state:

    <bos>State:\n{state}\n\nQuestion ({kind}):\n{question}\n\nOption:\n{option}

The readout is the final hidden state at the last token of each row (the option is never cut and sits at the end); a
linear layer turns it into one score per option and the scores of one decision are softmaxed together.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from dataclasses import replace
from typing import Optional

import torch

from .schema import Decision

KIND_ID = {"noul": 0, "choice": 1, "score": 2}
LETTERS = [chr(ord("A") + i) for i in range(26)]
MAX_OPTION_TOKENS = 160
CHAT_HEAD = "<|turn>user\nState:\n"            # after <bos>
CHAT_END = "<turn|>\n<|turn>model\n"           # closes the user turn and opens the model turn (generation prompt)
MIDDLE = " […] "


def mode_for(model_type: str) -> str:
    """Formatter mode of a model type (the letter baseline is the default)."""
    return {"slot": "slot", "scalar": "scalar", "span": "span"}.get(model_type, "letter")


@dataclass
class Encoded:
    input_ids: list[int]       # the sequence; for the scalar mode the longest row
    slot_pos: list[int]        # marker positions (slot model) in presented order
    perm: list[int]            # presented option j is original option perm[j]
    n_options: int
    rows: Optional[list[list[int]]] = None   # scalar mode: one sequence per option, in presented order
    spans: Optional[list[tuple[int, int]]] = None   # span mode: [start, end) token range of each option's text


class Formatter:
    def __init__(self, tok, mode: str, max_len: int, marker_id: int, chat: bool = False):
        assert mode in ("slot", "letter", "scalar", "span")
        self.tok, self.mode, self.max_len, self.marker_id = tok, mode, max_len, marker_id
        self.chat = chat or mode == "letter"   # the letter baseline is always in chat format
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
        if self.mode == "scalar":
            head = [self.bos] + self._ids(CHAT_HEAD if self.chat else "State:\n")
            mid = self._ids(f"\n\nQuestion ({d.kind}):\n{d.question.strip()}\n\nOption:\n")
            end = self._ids(CHAT_END) if self.chat else []
            tails = [mid + self._ids(o)[:MAX_OPTION_TOKENS] + end for o in opts]
            budget = self.max_len - len(head) - max(len(t) for t in tails)
            if budget < 16:  # question + longest option leave no room for any state: drop (counted by caller)
                return None
            state_ids = self._truncate_state(self._ids(d.state.strip()), budget)
            rows = [head + state_ids + t for t in tails]
            return Encoded(max(rows, key=len), [], perm, k, rows)
        body_lens: list[int] = []
        if self.mode in ("slot", "span"):
            head = [self.bos] + self._ids(CHAT_HEAD if self.chat else "State:\n")
            mid = self._ids(f"\n\nQuestion ({d.kind}):\n{d.question.strip()}\n\nOptions:\n")
            opt_ids = []
            for j, o in enumerate(opts):
                # chat: the template trims the user content, so the last option has no trailing newline
                newline = not self.chat or j < k - 1
                if self.mode == "span":   # the option text and its separator are tokenized apart: the span is the text
                    text = self._ids(o)[:MAX_OPTION_TOKENS]
                    body_lens.append(len(text))
                    opt_ids.append(text + (self._ids("\n") if newline else []))
                else:
                    body = self._ids(f" {o}" + ("\n" if newline else ""))[:MAX_OPTION_TOKENS]
                    opt_ids.append([self.marker_id] + body)
            end = self._ids(CHAT_END) if self.chat else []
            tail_len = len(mid) + sum(len(x) for x in opt_ids) + len(end)
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
        slot_pos, spans = [], []
        for j, oi in enumerate(opt_ids):
            if self.mode == "slot":
                slot_pos.append(len(ids))
            if self.mode == "span":
                spans.append((len(ids), len(ids) + body_lens[j]))
            ids += oi
        ids += end   # letter: the answer instruction and generation prompt; slot/span in chat format: the generation prompt
        return Encoded(ids, slot_pos, perm, k, None, spans or None)


def make_perm(d: Decision, rng: random.Random, augment: bool) -> list[int]:
    k = len(d.options)
    perm = list(range(k))
    if not augment:
        return perm
    if d.kind == "score":
        return perm[::-1] if rng.random() < 0.5 else perm  # ordering semantics preserved up to reversal
    rng.shuffle(perm)
    return perm


def subsample_negatives(d: Decision, perm: list[int], max_options: int, rng: random.Random):
    """Training only, scalar mode, choice decisions (D33): keep every option with target mass and a random subset of
    the others (at least one), so that a decision does not become dozens of rows. Returns the reduced decision in
    presented order, its identity permutation and the sampled-softmax offsets: a kept negative gets
    log(#negatives / #kept negatives) added to its logit during training, which keeps the normaliser unbiased."""
    pos = [j for j, i in enumerate(perm) if d.target[i] > 0]
    neg = [j for j, i in enumerate(perm) if d.target[i] <= 0]
    m = max(1, max_options - len(pos)) if neg else 0
    kept_neg = neg if len(neg) <= m else rng.sample(neg, m)
    keep = sorted(pos + kept_neg)
    corr = math.log(len(neg) / len(kept_neg)) if kept_neg else 0.0
    negset = set(kept_neg)
    offsets = [corr if j in negset else 0.0 for j in keep]
    d2 = replace(d, options=[d.options[perm[j]] for j in keep], target=[d.target[perm[j]] for j in keep],
                 option_values=None)
    return d2, list(range(len(keep))), offsets


class Collator:
    """Turns (Decision, seed) pairs into padded tensors. Used as DataLoader collate_fn.

    `max_train_options` (scalar mode, training): cap on the rows per choice decision, see `subsample_negatives`."""

    def __init__(self, formatter: Formatter, augment: bool, pad_id: int = 0, max_train_options: Optional[int] = None):
        self.f, self.augment, self.pad_id = formatter, augment, pad_id
        self.max_train_options = max_train_options

    def __call__(self, items):
        encs, decs, offs, drop = [], [], [], 0
        for it in items:
            d, seed = it if isinstance(it, tuple) else (it, 0)
            rng = random.Random(seed)
            perm, off = make_perm(d, rng, self.augment), None
            if (self.f.mode == "scalar" and self.augment and self.max_train_options and d.kind == "choice"
                    and len(d.options) > self.max_train_options):
                d, perm, off = subsample_negatives(d, perm, self.max_train_options, rng)
            e = self.f.encode(d, perm)
            if e is None:
                drop += 1
                continue
            encs.append(e)
            decs.append(d)
            offs.append(off or [0.0] * len(perm))
        if not encs:  # every item in this batch was too long: caller skips it
            return None, [], drop
        return self.pack(encs, decs, offs), decs, drop

    def pack_rows(self, encs: list[Encoded], decs: list[Decision], offsets=None) -> dict:
        """Scalar mode: one row per (decision, option). `row_dec` / `row_opt` map rows back to [B, K]."""
        B = len(encs)
        K = max(e.n_options for e in encs)
        rows = [(b, j, r) for b, e in enumerate(encs) for j, r in enumerate(e.rows)]
        T = max(len(r) for _, _, r in rows)
        R = len(rows)
        ids = torch.full((R, T), self.pad_id, dtype=torch.long)
        att = torch.zeros((R, T), dtype=torch.long)
        last = torch.zeros(R, dtype=torch.long)
        row_dec = torch.zeros(R, dtype=torch.long)
        row_opt = torch.zeros(R, dtype=torch.long)
        for i, (b, j, r) in enumerate(rows):
            ids[i, : len(r)] = torch.tensor(r)
            att[i, : len(r)] = 1
            last[i], row_dec[i], row_opt[i] = len(r) - 1, b, j
        valid = torch.zeros((B, K), dtype=torch.bool)
        target = torch.zeros((B, K), dtype=torch.float32)
        offset = torch.zeros((B, K), dtype=torch.float32)
        kind = torch.zeros(B, dtype=torch.long)
        for b, (e, d) in enumerate(zip(encs, decs)):
            valid[b, : e.n_options] = True
            target[b, : e.n_options] = torch.tensor([d.target[i] for i in e.perm], dtype=torch.float32)
            if offsets is not None:
                offset[b, : e.n_options] = torch.tensor(offsets[b], dtype=torch.float32)
            kind[b] = KIND_ID[d.kind]
        return {"input_ids": ids, "attention_mask": att, "token_type_ids": att.clone(), "last_pos": last,
                "row_dec": row_dec, "row_opt": row_opt, "valid": valid, "target": target, "kind": kind,
                "logit_offset": offset, "perm": [e.perm for e in encs]}

    def pack(self, encs: list[Encoded], decs: list[Decision], offsets=None) -> dict:
        if encs and encs[0].rows is not None:
            return self.pack_rows(encs, decs, offsets)
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
        batch = {
            "input_ids": ids, "attention_mask": att, "token_type_ids": att.clone(), "slot_pos": slot_pos,
            "valid": valid, "target": target, "last_pos": last, "kind": kind,
            "perm": [e.perm for e in encs],
        }
        if encs and encs[0].spans is not None:   # span mode: token range of every option's text
            batch["span_start"] = torch.zeros((B, K), dtype=torch.long)
            batch["span_end"] = torch.zeros((B, K), dtype=torch.long)
            for b, e in enumerate(encs):
                for j, (s, t) in enumerate(e.spans):
                    batch["span_start"][b, j], batch["span_end"][b, j] = s, t
        return batch
