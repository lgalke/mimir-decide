"""Run a decision model over records (identity option order) and return per-record logits."""
from __future__ import annotations

import random

import numpy as np
import torch

from .formatting import Collator, Formatter, make_perm
from .schema import Decision


def device_auto() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def unpermute(row: np.ndarray, perm: list[int]) -> np.ndarray:
    """Logits for options presented in order `perm` -> logits in the decision's original option order."""
    out = np.empty_like(row)
    for j, orig in enumerate(perm):
        out[orig] = row[j]
    return out


@torch.no_grad()
def predict_logits(model, tok, meta: dict, decs: list[Decision], batch_size: int = 16, device: str = "cpu",
                   autocast: bool = False, max_len: int | None = None, perm_seed: int | None = None):
    """Returns (logits list aligned with kept decs, kept decs). Records whose question+options do not fit are
    skipped (reported by the caller via len difference). With `perm_seed`, options are presented in a seeded random
    order (score decisions: identity or reversed) and the logits are mapped back to the original order, which
    measures robustness to option position."""
    mode = "slot" if meta["model_type"] == "slot" else "letter"
    fmt = Formatter(tok, mode, max_len or meta["max_len"], meta["marker_id"])
    col = Collator(fmt, augment=False, pad_id=tok.pad_token_id or 0)
    order = sorted(range(len(decs)), key=lambda i: len(decs[i].state) + sum(map(len, decs[i].options)))
    out_logits, out_decs = {}, {}
    model.eval()
    for s in range(0, len(order), batch_size):
        chunk = [decs[i] for i in order[s : s + batch_size]]
        enc = [(i, fmt.encode(decs[i], None if perm_seed is None else
                              make_perm(decs[i], random.Random(perm_seed * 1000003 + i), True)))
               for i in order[s : s + batch_size]]
        keep = [(i, e) for i, e in enc if e is not None]
        if not keep:
            continue
        batch = col.pack([e for _, e in keep], [decs[i] for i, _ in keep])
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=autocast):
            lg = model(batch).float().cpu().numpy()
        for (i, e), row in zip(keep, lg):
            out_logits[i] = unpermute(row[: e.n_options], e.perm)
            out_decs[i] = decs[i]
        del chunk
    idx = sorted(out_logits)
    return [out_logits[i] for i in idx], [out_decs[i] for i in idx]
