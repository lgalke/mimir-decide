"""Worst-case memory and speed probe: how large can the micro-batch be when every sequence has `max_len` tokens?

    python -m mimir_decide.mem_probe --config configs/train_baseline_v1.yaml \\
        --batch_sizes 1 2 4 6 8 12 16 24 32 --checkpointing true false

For each checkpointing setting the model is built as in training (fp32 weights, bf16 autocast, AdamW with the same
parameter groups), then one batch of identical full-length decisions (the state is long enough that the formatter cuts
it to exactly `max_len` tokens) goes through two complete optimizer steps for every batch size. The probe reports the
peak allocated and reserved memory, the time of the second step and examples per second, and stops at the first
out-of-memory error of a setting. Real batches are shorter on average, so this is the upper bound for a fixed micro-batch
size: batches are bucketed by length, and the longest ones occur regularly (one batch in a chunk of 50 holds the
longest sequences). Nothing is saved.
"""
from __future__ import annotations

import argparse
import gc
import json
import time

import torch
import yaml

from . import paths
from .formatting import Collator, Formatter, mode_for
from .inference import cap_gpu_memory, device_auto
from .model import build_model, decision_loss
from .schema import Decision
from .train import backbone_groups


def full_length_decision(options: int = 4) -> Decision:
    n = max(options, 2)
    t = [1.0] + [0.0] * (n - 1)
    return Decision(id="probe", case_id="probe", group_id="probe", dataset="probe", source="probe/probe",
                    split="train", language="en", license="n/a", kind="choice",
                    state="lorem ipsum dolor sit amet " * 6000, question="Which option?",
                    options=[f"option {i}" for i in range(n)], target=t)


def sweep(run_step, batch_sizes, device: str) -> list[dict]:
    """run_step(B) -> seconds of the second optimizer step. Stops at the first out-of-memory error."""
    rows = []
    cuda = device.startswith("cuda")
    for B in batch_sizes:
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        try:
            t = run_step(B)
        except torch.OutOfMemoryError:
            rows.append({"batch_size": B, "ok": False})
            break
        rows.append({"batch_size": B, "ok": True, "step_s": round(t, 3), "ex_per_s": round(B / t, 2),
                     "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1) if cuda else None,
                     "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2 ** 30, 1) if cuda else None})
    return rows


def make_step_fn(model, opt, groups, collate, decision: Decision, device: str, emd_lambda: float = 0.5, clip: float = 1.0):
    enc = collate.f.encode(decision)
    assert enc is not None, "probe decision does not fit"

    def one(B: int):
        batch = collate.pack([enc] * B, [decision] * B)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            logits = model(batch)
        loss, _, _ = decision_loss(logits.float(), batch, emd_lambda)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

    def run_step(B: int) -> float:
        try:
            one(B)                          # first step allocates the optimizer state
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.time()
            one(B)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            return time.time() - t0
        except torch.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            raise

    return run_step


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides key=value (YAML-parsed), as in train.py")
    ap.add_argument("--batch_sizes", nargs="*", type=int, default=[1, 2, 4, 6, 8, 12, 16, 24, 32])
    ap.add_argument("--checkpointing", nargs="*", default=["true", "false"], help="gradient checkpointing settings to try")
    ap.add_argument("--options", type=int, default=4, help="options per probe decision (scalar model: rows per decision)")
    ap.add_argument("--effective", type=int, default=None, help="effective batch for the suggested pairs (default batch_size*grad_accum)")
    a = ap.parse_args(argv)
    cfg = paths.load_config(a.config)
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    device = cfg.get("device") or device_auto()
    cap_gpu_memory(device, cfg.get("gpu_memory_fraction"))
    effective = a.effective or cfg["batch_size"] * cfg.get("grad_accum", 1)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer") or cfg["base_model"],
                                        revision=cfg.get("tokenizer_revision", cfg.get("revision")))
    marker_id = tok.convert_tokens_to_ids("<unused0>")
    mode = mode_for(cfg["model_type"])
    chat = bool(cfg.get("chat_format", cfg["model_type"] in ("scalar", "span", "clef")))
    collate = Collator(Formatter(tok, mode, cfg["max_len"], marker_id, chat), augment=False, pad_id=tok.pad_token_id or 0)
    dec = full_length_decision(a.options)
    results = {}
    for ck in a.checkpointing:
        c = {**cfg, "gradient_checkpointing": ck.lower() in ("1", "true", "yes")}
        model = build_model(c, tok, device, getattr(torch, c.get("param_dtype", "float32")))
        model.train()
        groups, _ = backbone_groups(model.lm, c)
        new = model.new_parameters()
        if new:
            groups.append({"params": new, "lr": float(c["lr_head"]), "weight_decay": 0.0})
        opt = torch.optim.AdamW(groups, lr=float(c["lr_backbone"]), betas=(0.9, 0.95), fused=device.startswith("cuda"))
        rows = sweep(make_step_fn(model, opt, groups, collate, dec, device), a.batch_sizes, device)
        results[f"gradient_checkpointing={c['gradient_checkpointing']}"] = rows
        print(f"\n== {cfg['model_type']} model, max_len {cfg['max_len']}, gradient_checkpointing={c['gradient_checkpointing']}")
        for r in rows:
            print("  ", json.dumps(r))
        ok = [r["batch_size"] for r in rows if r["ok"]]
        fit = [b for b in ok if effective % b == 0]
        print(f"   largest batch that fits: {max(ok) if ok else 'none'}"
              + (f"; with effective batch {effective}: batch_size={max(fit)} grad_accum={effective // max(fit)}" if fit else ""))
        del model, opt, groups
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    print("\n" + json.dumps(results))


if __name__ == "__main__":
    main()
