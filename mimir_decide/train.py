"""Single-GPU training loop for Mimir-Decide (slot model or letter baseline).

Reads ONLY train.parquet (+ validation.parquet for periodic eval / best-checkpoint selection). Test and held-out
task files are never opened here, and `assert_trainable_path` refuses them even if configured by mistake.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from .formatting import Collator, Formatter
from .inference import device_auto, predict_logits
from .metrics import evaluate
from .model import build_model, decision_loss, save_model
from .schema import Decision, read_parquet

FORBIDDEN_PARTS = {"eval", "heldout_tasks"}


def assert_trainable_path(p: Path) -> Path:
    p = Path(p).expanduser().resolve()
    if FORBIDDEN_PARTS & set(p.parts) or p.name.startswith("test") or p.name.startswith("heldout"):
        raise PermissionError(f"refusing to read non-train data in training: {p}")
    return p


class _Pairs(Dataset):
    def __init__(self, items: list[Decision]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, key):
        idx, seed = key
        return self.items[idx], seed


class LengthBucketBatches:
    """Shuffled batches; within mega-chunks of 50 batches, sort by text length to reduce padding."""

    def __init__(self, items: list[Decision], batch_size: int, seed: int, epoch: int, mega: int = 50):
        rng = random.Random(seed * 1000003 + epoch)
        idx = list(range(len(items)))
        rng.shuffle(idx)
        size = lambda i: len(items[i].state) + sum(len(o) for o in items[i].options)
        batches = []
        for s in range(0, len(idx), batch_size * mega):
            chunk = sorted(idx[s : s + batch_size * mega], key=size)
            batches += [chunk[j : j + batch_size] for j in range(0, len(chunk), batch_size)]
        rng.shuffle(batches)
        self.batches = [[(i, rng.randrange(2**31)) for i in b] for b in batches]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def lr_factor(step: int, total: int, warmup: int, floor: float = 0.1) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides key=value (yaml-parsed)")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config))
    for kv in a.set:
        k, v = kv.split("=", 1)
        val = yaml.safe_load(v)
        if isinstance(val, str):
            try:  # YAML 1.1 parses "1e-3" as a string
                val = float(val)
            except ValueError:
                pass
        cfg[k] = val

    seed = cfg.get("seed", 0)
    random.seed(seed)
    torch.manual_seed(seed)
    device = cfg.get("device") or device_auto()
    param_dtype = getattr(torch, cfg.get("param_dtype", "float32"))
    use_autocast = cfg.get("autocast", device == "cuda")
    data_dir = Path(cfg["data_dir"]).expanduser()
    run_dir = Path(cfg["run_dir"]).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)

    train = read_parquet(assert_trainable_path(data_dir / "train.parquet"))
    val = read_parquet(assert_trainable_path(data_dir / "validation.parquet"))
    assert all(d.split == "train" for d in train), "train.parquet contains non-train records"
    if cfg.get("max_train_examples"):
        train = train[: cfg["max_train_examples"]]
    val = val[: cfg.get("eval_max_examples", 2000)]
    print(f"train={len(train)} val={len(val)} device={device} model_type={cfg['model_type']}")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer") or cfg["base_model"], revision=cfg.get("tokenizer_revision", cfg.get("revision")))
    marker_id = tok.convert_tokens_to_ids("<unused0>")
    model = build_model(cfg, tok, device, param_dtype)
    mode = "slot" if cfg["model_type"] == "slot" else "letter"
    max_len = cfg["max_len"]
    fmt = Formatter(tok, mode, max_len, marker_id)
    collate = Collator(fmt, augment=True, pad_id=tok.pad_token_id or 0)
    meta = {"model_type": cfg["model_type"], "max_len": max_len, "marker_id": marker_id,
            "base_model": cfg["base_model"], "revision": cfg.get("revision")}

    backbone = [p for p in model.lm.parameters() if p.requires_grad]
    new = model.new_parameters()
    groups = ([{"params": backbone, "lr": float(cfg["lr_backbone"]), "weight_decay": cfg.get("weight_decay", 0.01)}]
              if backbone else [])  # empty for head-only probes (freeze_backbone)
    if new:
        groups.append({"params": new, "lr": float(cfg["lr_head"]), "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), fused=(device == "cuda"))
    base_lrs = [g["lr"] for g in opt.param_groups]
    n_train = sum(p.numel() for g in groups for p in g["params"])
    print(f"trainable params: {n_train/1e6:.1f}M (backbone {sum(p.numel() for p in backbone)/1e6:.1f}M)")

    bs, accum = cfg["batch_size"], cfg.get("grad_accum", 1)
    steps_per_epoch = math.ceil(math.ceil(len(train) / bs) / accum)
    total_steps = cfg.get("max_steps") or steps_per_epoch * cfg.get("epochs", 1)
    warmup = max(1, int(cfg.get("warmup_frac", 0.01) * total_steps))
    shutil.copy(a.config, run_dir / "config.yaml")
    (run_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=1, default=str))
    mm = data_dir / "mixture_manifest.json"
    if mm.exists():
        shutil.copy(mm, run_dir / "mixture_manifest.json")
        meta["mixture_manifest_sha256"] = hashlib.sha256(mm.read_bytes()).hexdigest()
    log = open(run_dir / "train_log.jsonl", "a")

    def do_eval(step):
        model.eval()
        lg, kept = predict_logits(model, tok, meta, val, cfg.get("eval_batch_size", 16), device, use_autocast)
        model.train()
        r = evaluate(lg, kept)["all"]
        rec = {"step": step, "val_nll": r["nll"], "val_acc": r["acc"], "val_ece": r["ece"], "val_n": r["n"]}
        print(rec)
        log.write(json.dumps({"eval": rec}) + "\n")
        log.flush()
        return r["nll"]

    model.train()
    step, micro, best, t0, skipped, seen = 0, 0, float("inf"), time.time(), 0, 0
    ep = 0
    done = False
    while not done:
        loader = DataLoader(_Pairs(train), batch_sampler=LengthBucketBatches(train, bs, seed, ep),
                            collate_fn=collate, num_workers=cfg.get("num_workers", 2),
                            persistent_workers=False)
        for batch, decs, dropped in loader:
            skipped += dropped
            if not decs:
                continue
            seen += len(decs)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_autocast):
                logits = model(batch)
            loss, ce, emd = decision_loss(logits.float(), batch, cfg.get("emd_lambda", 0.5))
            (loss / accum).backward()
            micro += 1
            if micro % accum:
                continue
            f = lr_factor(step, total_steps, warmup)
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * f
            gn = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], cfg.get("grad_clip", 1.0))
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.get("log_every", 10) == 0 or step == 1:
                rec = {"step": step, "loss": loss.item(), "ce": ce.mean().item(), "grad_norm": float(gn),
                       "lr": opt.param_groups[0]["lr"], "elapsed_s": round(time.time() - t0, 1), "skipped": skipped,
                       "ex_per_s": round(seen / max(1e-9, time.time() - t0), 2),
                       "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else None}
                print(rec)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            if step % cfg.get("eval_every", 1000) == 0:
                nll = do_eval(step)
                if nll < best:
                    best = nll
                    save_model(model, tok, run_dir / "best", {**meta, "step": step, "val_nll": nll})
            if step >= total_steps:
                done = True
                break
        ep += 1
    nll = do_eval(step)
    if nll < best:
        best = nll
        save_model(model, tok, run_dir / "best", {**meta, "step": step, "val_nll": nll})
    save_model(model, tok, run_dir / "final", {**meta, "step": step, "val_nll": nll})
    print(f"done: steps={step} best_val_nll={best:.4f} skipped_too_long={skipped}")


if __name__ == "__main__":
    main()
