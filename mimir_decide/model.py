"""Decision models on top of HRM-Text (transformers' built-in `hrm_text`).

SlotDecisionModel  - one logit per option, read from the final H-state at each option marker.
LetterBaseline     - plain LM next-token logits over option letters (A, B, ...), same data/loss/metrics.

Both return logits [B, K] with invalid option slots masked, so loss/metrics/calibration are shared.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.hrm_text.modeling_hrm_text import HrmTextForCausalLM

from .formatting import LETTERS

NEG = -1e9


class SlotDecisionModel(nn.Module):
    kind = "slot"

    def __init__(self, lm: HrmTextForCausalLM, freeze_input_embeddings: bool = True):
        super().__init__()
        self.lm = lm
        D = lm.config.hidden_size
        for p in lm.lm_head.parameters():  # kept in checkpoints for reuse, not trained
            p.requires_grad = False
        if freeze_input_embeddings:
            lm.model.embed_tokens.weight.requires_grad = False
        w = lm.model.embed_tokens.weight.detach()
        self.marker = nn.Parameter(w.float().mean(0).to(w.dtype).clone())  # unscaled embedding space
        self.head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1))
        nn.init.normal_(self.head[1].weight, std=0.02)
        nn.init.zeros_(self.head[1].bias)
        self.log_scale = nn.Parameter(torch.zeros(3))  # per kind (noul, choice, score)

    def new_parameters(self):
        return [self.marker, *self.head.parameters(), self.log_scale]

    def forward(self, batch: dict) -> torch.Tensor:
        ids = batch["input_ids"]
        marker_mask = torch.zeros_like(ids, dtype=torch.bool)
        marker_mask.scatter_(1, batch["slot_pos"], batch["valid"])
        emb = self.lm.model.embed_tokens(ids)
        emb = torch.where(marker_mask.unsqueeze(-1), self.marker.to(emb.dtype), emb)
        out = self.lm.model(inputs_embeds=emb, attention_mask=batch["attention_mask"],
                            token_type_ids=batch["token_type_ids"])
        h = out.last_hidden_state
        idx = batch["slot_pos"].unsqueeze(-1).expand(-1, -1, h.size(-1))
        hs = torch.gather(h, 1, idx)  # [B,K,D]
        logits = self.head(hs.float()).squeeze(-1) * torch.exp(self.log_scale)[batch["kind"]].unsqueeze(-1)
        return logits.masked_fill(~batch["valid"], NEG)

    def extra_state(self) -> dict:
        return {"marker": self.marker.detach().cpu(), "head": self.head.state_dict(),
                "log_scale": self.log_scale.detach().cpu()}

    def load_extra(self, st: dict) -> None:
        with torch.no_grad():
            self.marker.copy_(st["marker"])
            self.log_scale.copy_(st["log_scale"])
        self.head.load_state_dict(st["head"])


class LetterBaseline(nn.Module):
    kind = "letter"

    def __init__(self, tok, lm: HrmTextForCausalLM, freeze_input_embeddings: bool = True,
                 train_lm_head: bool = False):
        super().__init__()
        self.lm = lm
        if freeze_input_embeddings:
            lm.model.embed_tokens.weight.requires_grad = False
        for p in lm.lm_head.parameters():
            p.requires_grad = train_lm_head
        ids = [tok(l, add_special_tokens=False)["input_ids"] for l in LETTERS]
        assert all(len(i) == 1 for i in ids), "letters must be single tokens"
        self.register_buffer("letter_ids", torch.tensor([i[0] for i in ids]), persistent=False)

    def new_parameters(self):
        return []

    def forward(self, batch: dict) -> torch.Tensor:
        out = self.lm.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                            token_type_ids=batch["token_type_ids"])
        h = out.last_hidden_state
        h = h[torch.arange(h.size(0), device=h.device), batch["last_pos"]]  # [B,D]
        w = self.lm.lm_head.weight[self.letter_ids]  # [26,D]
        logits = (h @ w.t().to(h.dtype)).float()
        K = batch["valid"].size(1)
        return logits[:, :K].masked_fill(~batch["valid"], NEG)

    def extra_state(self) -> dict:
        return {}

    def load_extra(self, st: dict) -> None:
        pass


# ---------------------------------------------------------------------------------------------- losses
def decision_loss(logits: torch.Tensor, batch: dict, emd_lambda: float = 0.5):
    """Soft cross-entropy for every kind + squared-CDF distance (EMD^2) for score decisions."""
    logp = F.log_softmax(logits, dim=-1)
    target = batch["target"]
    ce = -(target * logp).sum(-1)
    p = logp.exp()
    emd = ((p.cumsum(-1) - target.cumsum(-1)) ** 2).sum(-1)
    is_score = (batch["kind"] == 2).float()
    per = ce + emd_lambda * is_score * emd
    return per.mean(), ce.detach(), emd.detach()


# ---------------------------------------------------------------------------------------------- build/save/load
def build_model(cfg: dict, tok, device: str, dtype: torch.dtype = torch.float32):
    from transformers import AutoConfig

    base, rev = cfg["base_model"], cfg.get("revision")
    config = AutoConfig.from_pretrained(base, revision=rev)
    if cfg.get("L_bp_cycles") is not None:
        config.L_bp_cycles = list(cfg["L_bp_cycles"])
    lm = HrmTextForCausalLM.from_pretrained(base, revision=rev, config=config, attn_implementation="sdpa",
                                            dtype=dtype)
    if cfg.get("gradient_checkpointing"):
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        lm.config.use_cache = False
    fe = cfg.get("freeze_input_embeddings", True)
    if cfg["model_type"] == "slot":
        m = SlotDecisionModel(lm, fe)
    elif cfg["model_type"] == "letter":
        m = LetterBaseline(tok, lm, fe, cfg.get("train_lm_head", False))
    else:
        raise ValueError(cfg["model_type"])
    return m.to(device)


def save_model(model, tok, path, meta: dict) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in model.lm.state_dict().items()}
    model.lm.save_pretrained(path / "lm", state_dict=sd)  # bf16 inference weights
    torch.save(model.extra_state(), path / "extra.pt")
    tok.save_pretrained(path / "tokenizer")
    (path / "decision_meta.json").write_text(json.dumps(meta, indent=1))


def load_model(path, device: str, dtype: torch.dtype = torch.bfloat16):
    from transformers import AutoTokenizer

    path = Path(path)
    meta = json.loads((path / "decision_meta.json").read_text())
    tok = AutoTokenizer.from_pretrained(path / "tokenizer")
    lm = HrmTextForCausalLM.from_pretrained(path / "lm", attn_implementation="sdpa", dtype=dtype)
    if meta["model_type"] == "slot":
        m = SlotDecisionModel(lm)
    else:
        m = LetterBaseline(tok, lm)
    st = torch.load(path / "extra.pt", map_location="cpu")
    if st:
        m.load_extra(st)
    return m.to(device).eval(), tok, meta
