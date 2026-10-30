"""Decision models on top of HRM-Text (transformers' built-in `hrm_text`).

SlotDecisionModel  - one logit per option, read from the final H-state at each option marker.
LetterBaseline     - plain LM next-token logits over option letters (A, B, ...), same data/loss/metrics.

Both return logits [B, K] with invalid option slots masked, so loss/metrics/calibration are shared.
"""
from __future__ import annotations

import json
import math
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

    def __init__(self, lm: HrmTextForCausalLM, freeze_input_embeddings: bool = True, freeze_backbone: bool = False):
        super().__init__()
        self.lm = lm
        self.frozen_backbone = freeze_backbone  # head-only probe: backbone and marker fixed, no backward through the backbone
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
        if freeze_backbone:
            for p in lm.parameters():
                p.requires_grad = False
            self.marker.requires_grad = False  # stays at its initialisation (mean embedding)

    def new_parameters(self):
        return [p for p in (self.marker, *self.head.parameters(), self.log_scale) if p.requires_grad]

    def forward(self, batch: dict) -> torch.Tensor:
        ids = batch["input_ids"]
        marker_mask = torch.zeros_like(ids, dtype=torch.bool)
        marker_mask.scatter_(1, batch["slot_pos"], batch["valid"])
        emb = self.lm.model.embed_tokens(ids)
        emb = torch.where(marker_mask.unsqueeze(-1), self.marker.to(emb.dtype), emb)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen_backbone):
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


CLEF_DEFAULTS = {"width": 512, "routing_layers": 2, "layers": 2, "heads": 8, "feedforward": 2048, "dropout": 0.0}


class EvidenceRoutingLayer(nn.Module):
    """Cross-attention of the option queries to the whole sequence (the evidence), then a feed-forward block."""

    def __init__(self, width: int, heads: int, feedforward: int, dropout: float = 0.0):
        super().__init__()
        self.query_norm, self.memory_norm, self.feedforward_norm = nn.LayerNorm(width), nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.feedforward = nn.Sequential(nn.Linear(width, feedforward), nn.GELU(), nn.Dropout(dropout),
                                         nn.Linear(feedforward, width), nn.Dropout(dropout))

    def forward(self, queries: torch.Tensor, memory: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        mem = self.memory_norm(memory)
        routed, _ = self.attention(self.query_norm(queries), mem, mem, key_padding_mask=pad, need_weights=False)
        queries = queries + routed
        return queries + self.feedforward(self.feedforward_norm(queries))


class ClefHead(nn.Module):
    """Scoring head in the design of Cloudflare's Clef (joint_schema_model.py), for one question per sequence (D38).

    An option is described by (a) the mean of the final hidden states over its text, (b) a lexical vector, the mean of
    the output-embedding rows of its tokens, and (c) the mean state of the question text. Two cross-attention layers
    route evidence from the whole sequence to the options; a small decoder builds a field vector for the question; the
    score of an option is `prior + sigmoid(gate) * joint` with a lexical prior (cosine between the option's output
    embedding and the question-plus-last-token state) and a joint term (cosine plus a small MLP of field and option)."""

    def __init__(self, hidden_size: int, width: int, routing_layers: int, layers: int, heads: int, feedforward: int,
                 dropout: float = 0.0):
        super().__init__()
        self.width = width
        self.hidden_norm = nn.LayerNorm(hidden_size)
        proj = lambda: nn.Linear(hidden_size, width, bias=False)
        self.memory_projection, self.question_projection, self.option_question_projection = proj(), proj(), proj()
        self.global_projection, self.option_context_projection, self.option_lexical_projection = proj(), proj(), proj()
        self.type_embedding = nn.Embedding(3, width)
        self.evidence_layers = nn.ModuleList([EvidenceRoutingLayer(width, heads, feedforward, dropout)
                                              for _ in range(routing_layers)])
        self.option_summary_norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(width, heads, feedforward, dropout, activation="gelu",
                                                                batch_first=True, norm_first=True) for _ in range(layers)])
        self.field_norm, self.option_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.residual_scorer = nn.Sequential(nn.Linear(width * 4, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1))
        self.prior_logit_scale = nn.Parameter(torch.zeros(()))
        self.joint_logit_scale = nn.Parameter(torch.zeros(()))
        self.residual_gate = nn.Parameter(torch.zeros(()))

    def forward(self, h, attention_mask, last_pos, q_start, q_end, span_start, span_end, kind, valid, lexical_tokens):
        B, T, _ = h.shape
        hn = self.hidden_norm(h)
        pad = attention_mask == 0                                                     # keys to ignore
        memory = self.memory_projection(hn)                                           # [B, T, W]
        glob = hn[torch.arange(B, device=h.device), last_pos]                         # [B, D] last token
        pos = torch.arange(T, device=h.device)[None, None, :]
        mq = ((pos >= q_start[:, None, None]) & (pos < q_end[:, None, None])).float()                 # [B, 1, T]
        qvec = torch.bmm(mq, hn).squeeze(1) / mq.sum(-1).clamp(min=1.0)                                 # [B, D]
        mo = ((pos >= span_start[..., None]) & (pos < span_end[..., None])).float()                     # [B, K, T]
        count = mo.sum(-1, keepdim=True).clamp(min=1.0)
        ctx = torch.bmm(mo, hn) / count                                               # [B, K, D] contextual option vector
        lex = torch.bmm(mo, lexical_tokens) / count                                   # [B, K, D] lexical option vector
        routed = (self.option_context_projection(ctx) + self.option_lexical_projection(lex)
                  + self.option_question_projection(qvec)[:, None, :])                # [B, K, W]
        for layer in self.evidence_layers:
            routed = layer(routed, memory, pad)
        base = self.question_projection(qvec)                                         # [B, W]
        route_scores = (routed * base[:, None, :]).sum(-1) / math.sqrt(self.width)
        weights = torch.softmax(route_scores.masked_fill(~valid, float("-inf")), dim=1)
        summary = (weights[..., None] * routed).sum(1)
        field = (base + self.option_summary_norm(summary) + self.global_projection(glob) + self.type_embedding(kind))[:, None, :]
        for layer in self.layers:
            field = layer(field, memory, memory_key_padding_mask=pad)
        field = self.field_norm(field[:, 0])                                          # [B, W]
        anchor = F.normalize(qvec + glob, dim=-1)
        prior = self.prior_logit_scale.clamp(max=math.log(100.0)).exp() * torch.einsum("bkd,bd->bk", F.normalize(lex, dim=-1), anchor)
        options = self.option_norm(routed)
        rep = field[:, None, :].expand_as(options)
        cosine = F.cosine_similarity(rep, options, dim=-1)
        residual = self.residual_scorer(torch.cat([rep, options, rep * options, (rep - options).abs()], dim=-1)).squeeze(-1)
        joint = self.joint_logit_scale.clamp(max=math.log(100.0)).exp() * cosine + residual
        return prior + torch.sigmoid(self.residual_gate) * joint


class ClefHeadModel(nn.Module):
    """Backbone plus the Clef-style head (D38); the backbone is trained or frozen as for the other models."""
    kind = "clef"

    def __init__(self, lm: HrmTextForCausalLM, freeze_input_embeddings: bool = True, freeze_backbone: bool = False,
                 clef: dict | None = None):
        super().__init__()
        self.lm = lm
        self.frozen_backbone = freeze_backbone
        self.clef_cfg = {**CLEF_DEFAULTS, **(clef or {})}
        for p in lm.lm_head.parameters():      # frozen: its rows are the lexical vectors of the options
            p.requires_grad = False
        if freeze_input_embeddings:
            lm.model.embed_tokens.weight.requires_grad = False
        self.head = ClefHead(lm.config.hidden_size, **self.clef_cfg)
        if freeze_backbone:
            for p in lm.parameters():
                p.requires_grad = False

    def new_parameters(self):
        return list(self.head.parameters())

    def forward(self, batch: dict) -> torch.Tensor:
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen_backbone):
            out = self.lm.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                token_type_ids=batch["token_type_ids"])
        with torch.no_grad():
            lexical = self.lm.lm_head.weight[batch["input_ids"]].float()               # [B, T, D] output embeddings
        with torch.autocast(device_type=out.last_hidden_state.device.type, enabled=False):
            logits = self.head(out.last_hidden_state.float(), batch["attention_mask"], batch["last_pos"],
                               batch["q_start"], batch["q_end"], batch["span_start"], batch["span_end"],
                               batch["kind"], batch["valid"], lexical)
        return logits.masked_fill(~batch["valid"], NEG)

    def extra_state(self) -> dict:
        return {"head": self.head.state_dict(), "cfg": self.clef_cfg}

    def load_extra(self, st: dict) -> None:
        self.head.load_state_dict(st["head"])


class SpanPoolModel(nn.Module):
    """Span readout (D35): all options in one sequence; the representation of an option is the mean of the final hidden
    states over the tokens of its own text; one head per kind (noul, choice, score) turns it into a score."""
    kind = "span"

    def __init__(self, lm: HrmTextForCausalLM, freeze_input_embeddings: bool = True, freeze_backbone: bool = False,
                 per_kind_heads: bool = True):
        super().__init__()
        self.lm = lm
        self.frozen_backbone = freeze_backbone
        self.per_kind_heads = per_kind_heads
        D = lm.config.hidden_size
        for p in lm.lm_head.parameters():
            p.requires_grad = False
        if freeze_input_embeddings:
            lm.model.embed_tokens.weight.requires_grad = False
        self.heads = nn.ModuleList([nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1)) for _ in range(3 if per_kind_heads else 1)])
        for h in self.heads:
            nn.init.normal_(h[1].weight, std=0.02)
            nn.init.zeros_(h[1].bias)
        self.log_scale = nn.Parameter(torch.zeros(3))  # only used with one shared head
        if freeze_backbone:
            for p in lm.parameters():
                p.requires_grad = False

    def new_parameters(self):
        ps = list(self.heads.parameters())
        return ps if self.per_kind_heads else ps + [self.log_scale]

    def forward(self, batch: dict) -> torch.Tensor:
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen_backbone):
            out = self.lm.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                token_type_ids=batch["token_type_ids"])
        h = out.last_hidden_state.float()                                            # [B, T, D]
        pos = torch.arange(h.size(1), device=h.device)[None, None, :]
        m = ((pos >= batch["span_start"][..., None]) & (pos < batch["span_end"][..., None])).float()   # [B, K, T]
        pooled = torch.bmm(m, h) / m.sum(-1, keepdim=True).clamp(min=1.0)           # [B, K, D] mean over the option text
        scores = torch.stack([hd(pooled).squeeze(-1) for hd in self.heads], dim=-1)   # [B, K, n_heads]
        if self.per_kind_heads:
            idx = batch["kind"][:, None, None].expand(-1, scores.size(1), 1)
            logits = scores.gather(-1, idx).squeeze(-1)
        else:
            logits = scores[..., 0] * torch.exp(self.log_scale)[batch["kind"]].unsqueeze(-1)
        return logits.masked_fill(~batch["valid"], NEG)

    def extra_state(self) -> dict:
        return {"heads": self.heads.state_dict(), "log_scale": self.log_scale.detach().cpu(),
                "per_kind_heads": self.per_kind_heads}

    def load_extra(self, st: dict) -> None:
        assert st.get("per_kind_heads", True) == self.per_kind_heads, "checkpoint and model differ in per_kind_heads"
        with torch.no_grad():
            self.log_scale.copy_(st["log_scale"])
        self.heads.load_state_dict(st["heads"])


class OptionScorerModel(nn.Module):
    """Scalar head per option (D33): every option is its own sequence; the final hidden state at its last token goes
    through LayerNorm -> Linear(D, 1) and the scores of one decision are scattered into [B, K] logits."""
    kind = "scalar"

    def __init__(self, lm: HrmTextForCausalLM, freeze_input_embeddings: bool = True, freeze_backbone: bool = False):
        super().__init__()
        self.lm = lm
        self.frozen_backbone = freeze_backbone
        D = lm.config.hidden_size
        for p in lm.lm_head.parameters():
            p.requires_grad = False
        if freeze_input_embeddings:
            lm.model.embed_tokens.weight.requires_grad = False
        self.head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1))
        nn.init.normal_(self.head[1].weight, std=0.02)
        nn.init.zeros_(self.head[1].bias)
        self.log_scale = nn.Parameter(torch.zeros(3))  # per kind (noul, choice, score)
        if freeze_backbone:
            for p in lm.parameters():
                p.requires_grad = False

    def new_parameters(self):
        return [p for p in (*self.head.parameters(), self.log_scale) if p.requires_grad]

    def forward(self, batch: dict) -> torch.Tensor:
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen_backbone):
            out = self.lm.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                token_type_ids=batch["token_type_ids"])
        h = out.last_hidden_state
        h = h[torch.arange(h.size(0), device=h.device), batch["last_pos"]]          # [R, D]
        s = self.head(h.float()).squeeze(-1)                                         # [R]
        B, K = batch["valid"].shape
        logits = s.new_zeros(B, K).index_put((batch["row_dec"], batch["row_opt"]), s)
        logits = logits * torch.exp(self.log_scale)[batch["kind"]].unsqueeze(-1)
        return logits.masked_fill(~batch["valid"], NEG)

    def extra_state(self) -> dict:
        return {"head": self.head.state_dict(), "log_scale": self.log_scale.detach().cpu()}

    def load_extra(self, st: dict) -> None:
        with torch.no_grad():
            self.log_scale.copy_(st["log_scale"])
        self.head.load_state_dict(st["head"])


class LetterBaseline(nn.Module):
    kind = "letter"

    def __init__(self, tok, lm: HrmTextForCausalLM, freeze_input_embeddings: bool = True,
                 train_lm_head: bool = False, train_letter_head: bool = False, freeze_backbone: bool = False):
        super().__init__()
        assert not (train_lm_head and freeze_backbone), "train_lm_head trains the backbone's output matrix; use train_letter_head"
        self.lm = lm
        self.frozen_backbone = freeze_backbone
        if freeze_input_embeddings:
            lm.model.embed_tokens.weight.requires_grad = False
        for p in lm.lm_head.parameters():
            p.requires_grad = train_lm_head
        ids = [tok(l, add_special_tokens=False)["input_ids"] for l in LETTERS]
        assert all(len(i) == 1 for i in ids), "letters must be single tokens"
        self.register_buffer("letter_ids", torch.tensor([i[0] for i in ids]), persistent=False)
        self.letter_head = None
        if train_letter_head:
            self.enable_letter_head()
        if freeze_backbone:
            for p in lm.parameters():
                p.requires_grad = False

    def enable_letter_head(self) -> None:
        """Trainable copy of the 26 letter rows of the LM head: the classifier head starts exactly at the LM head."""
        w = self.lm.lm_head.weight.detach()[self.letter_ids].float().clone()
        self.letter_head = nn.Parameter(w.to(self.letter_ids.device))

    def new_parameters(self):
        return [self.letter_head] if self.letter_head is not None else []

    def forward(self, batch: dict) -> torch.Tensor:
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen_backbone):
            out = self.lm.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                token_type_ids=batch["token_type_ids"])
        h = out.last_hidden_state
        h = h[torch.arange(h.size(0), device=h.device), batch["last_pos"]]  # [B,D]
        w = self.letter_head if self.letter_head is not None else self.lm.lm_head.weight[self.letter_ids]  # [26,D]
        logits = (h @ w.t().to(h.dtype)).float()
        K = batch["valid"].size(1)
        return logits[:, :K].masked_fill(~batch["valid"], NEG)

    def extra_state(self) -> dict:
        return {"letter_head": self.letter_head.detach().cpu()} if self.letter_head is not None else {}

    def load_extra(self, st: dict) -> None:
        if "letter_head" in st:
            if self.letter_head is None:
                self.enable_letter_head()
            with torch.no_grad():
                self.letter_head.copy_(st["letter_head"])


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
        m = SlotDecisionModel(lm, fe, cfg.get("freeze_backbone", False))
    elif cfg["model_type"] == "scalar":
        m = OptionScorerModel(lm, fe, cfg.get("freeze_backbone", False))
    elif cfg["model_type"] == "clef":
        m = ClefHeadModel(lm, fe, cfg.get("freeze_backbone", False), cfg.get("clef_head"))
    elif cfg["model_type"] == "span":
        m = SpanPoolModel(lm, fe, cfg.get("freeze_backbone", False), cfg.get("per_kind_heads", True))
    elif cfg["model_type"] == "letter":
        m = LetterBaseline(tok, lm, fe, cfg.get("train_lm_head", False), cfg.get("train_letter_head", False),
                           cfg.get("freeze_backbone", False))
    else:
        raise ValueError(cfg["model_type"])
    return m.to(device)


def save_model(model, tok, path, meta: dict) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if getattr(model, "frozen_backbone", False):  # head-only probe: the backbone is the untouched base model
        meta = {**meta, "backbone_from_hub": True}
    else:
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
    if (path / "lm").exists():
        lm = HrmTextForCausalLM.from_pretrained(path / "lm", attn_implementation="sdpa", dtype=dtype)
    else:  # zero-shot or head-only checkpoint: no backbone weights stored, load the untouched base model
        assert meta.get("zero_shot") or meta.get("backbone_from_hub"), \
            f"{path} has no lm/ weights and is neither a zero-shot nor a head-only checkpoint"
        lm = HrmTextForCausalLM.from_pretrained(meta["base_model"], revision=meta.get("revision"),
                                                attn_implementation="sdpa", dtype=dtype)
    if meta["model_type"] == "slot":
        m = SlotDecisionModel(lm)
    elif meta["model_type"] == "scalar":
        m = OptionScorerModel(lm)
    elif meta["model_type"] == "clef":
        m = ClefHeadModel(lm, clef=meta.get("clef_head"))
    elif meta["model_type"] == "span":
        m = SpanPoolModel(lm, per_kind_heads=meta.get("per_kind_heads", True))
    else:
        m = LetterBaseline(tok, lm)
    st = torch.load(path / "extra.pt", map_location="cpu")
    if st:
        m.load_extra(st)
    return m.to(device).eval(), tok, meta
