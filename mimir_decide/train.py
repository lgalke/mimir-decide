"""Single-GPU training loop for Mimir-Decide (slot model or letter baseline).

Reads ONLY train.parquet (+ validation.parquet for periodic eval / best-checkpoint selection). Test and held-out
task files are never opened here, and `assert_trainable_path` refuses them even if configured by mistake.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from . import paths
from .formatting import Collator, Formatter, mode_for
from .inference import cap_gpu_memory, device_auto, predict_logits
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

    def __init__(self, items: list[Decision], batch_size: int, seed: int, epoch: int, mega: int = 50, skip: int = 0):
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
        self.batches = self.batches[skip:]   # resume: the order is fixed by (seed, epoch), so skipping reproduces the run

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def lr_factor(step: int, total: int, warmup: int, floor: float = 0.1) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


def ema_decay_at(step: int, decay: float) -> float:
    """Decay of the weight average at an optimizer step: it warms up from about 0.1 towards `decay`, so that the first
    steps are not dominated by the starting point (the usual warm-up of EMA implementations)."""
    return min(decay, (1 + step) / (10 + step))


def ema_update(ema: list, params: list, decay: float) -> None:
    """ema <- decay * ema + (1 - decay) * params, in place, for lists of tensors of the same dtype."""
    torch._foreach_mul_(ema, decay)
    torch._foreach_add_(ema, [p.detach().to(e.dtype) for p, e in zip(params, ema)], alpha=1.0 - decay)


def swap_with_ema(params: list, ema: list) -> None:
    """Exchange the weights of the parameters with the averaged weights (no copy; calling it twice restores them).
    The optimizer keeps its state, because it refers to the Parameter objects."""
    for i, p in enumerate(params):
        tmp = p.data
        p.data = ema[i]
        ema[i] = tmp


@contextlib.contextmanager
def ema_scope(params: list, ema: list):
    """Inside the block the model holds the averaged weights; the training weights are restored on exit."""
    swap_with_ema(params, ema)
    try:
        yield
    finally:
        swap_with_ema(params, ema)


def cycle_dividers(lm) -> dict:
    """Learning-rate dividers per recurrent module ("Galke Poech dividers", D30).

    A module that is applied n times per forward pass, with gradients flowing through all applications, receives n
    summed gradient contributions, so its base learning rate is divided by n: L by the number of L-stack applications
    that carry gradients (sum of `L_bp_cycles`; 6 for H_cycles=2, L_cycles=3, L_bp_cycles=[3,3]), H by `H_cycles` (2).
    """
    inner = lm.model
    H = lm.config.H_cycles
    return {"L": max(1, sum(inner.L_bp_cycles_padded[:H])), "H": max(1, H)}


def backbone_groups(lm, cfg: dict):
    """AdamW parameter groups for the backbone. Default (flag off) is one group, exactly as in the earlier runs.
    With `lr_cycle_dividers: true` the L stack, the H stack and the remaining trainable weights get their own group.
    Returns (groups, dividers or None)."""
    lr, wd = float(cfg["lr_backbone"]), cfg.get("weight_decay", 0.01)
    trainable = [p for p in lm.parameters() if p.requires_grad]
    if not trainable:
        return [], None
    if not cfg.get("lr_cycle_dividers", False):
        return [{"params": trainable, "lr": lr, "weight_decay": wd}], None
    div = cycle_dividers(lm)
    in_L = [p for p in lm.model.L_module.parameters() if p.requires_grad]
    in_H = [p for p in lm.model.H_module.parameters() if p.requires_grad]
    taken = {id(p) for p in in_L + in_H}
    rest = [p for p in trainable if id(p) not in taken]
    groups = [{"params": ps, "lr": lr / d, "weight_decay": wd, "name": n}
              for ps, d, n in ((in_L, div["L"], "L"), (in_H, div["H"], "H"), (rest, 1, "other")) if ps]
    return groups, div


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides key=value (yaml-parsed)")
    a = ap.parse_args(argv)
    cfg = paths.load_config(a.config)
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
    cap_gpu_memory(device, cfg.get("gpu_memory_fraction"))
    param_dtype = getattr(torch, cfg.get("param_dtype", "float32"))
    use_autocast = cfg.get("autocast", device == "cuda")
    data_dir = Path(cfg["data_dir"]).expanduser()
    run_dir = Path(cfg["run_dir"]).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)

    train = read_parquet(assert_trainable_path(data_dir / cfg.get("train_file", "train.parquet")))  # v1: nested subsets
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
    mode = mode_for(cfg["model_type"])
    max_len = cfg["max_len"]
    chat = bool(cfg.get("chat_format", cfg["model_type"] in ("scalar", "span", "clef")))   # D34; slot-v0 and the probes were plain text
    fmt = Formatter(tok, mode, max_len, marker_id, chat)
    collate = Collator(fmt, augment=True, pad_id=tok.pad_token_id or 0, max_train_options=cfg.get("max_train_options"))
    meta = {"model_type": cfg["model_type"], "max_len": max_len, "marker_id": marker_id, "chat_format": chat,
            "per_kind_heads": bool(cfg.get("per_kind_heads", True)),
            **({"clef_head": model.clef_cfg} if hasattr(model, "clef_cfg") else {}),
            "base_model": cfg["base_model"], "revision": cfg.get("revision")}

    groups, dividers = backbone_groups(model.lm, cfg)  # empty for head-only probes (freeze_backbone)
    backbone = [p for g in groups for p in g["params"]]
    if dividers:
        meta["lr_dividers"] = dividers
        print(f"lr_cycle_dividers: L / {dividers['L']}, H / {dividers['H']} -> lr "
              + ", ".join(f"{g['name']}={g['lr']:.3g}" for g in groups))
    new = model.new_parameters()
    if new:
        groups.append({"params": new, "lr": float(cfg["lr_head"]), "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), fused=(device == "cuda"))
    base_lrs = [g["lr"] for g in opt.param_groups]
    n_train = sum(p.numel() for g in groups for p in g["params"])
    print(f"trainable params: {n_train/1e6:.1f}M (backbone {sum(p.numel() for p in backbone)/1e6:.1f}M)")

    ema_decay = cfg.get("ema_decay")      # weight average of the trainable parameters (D36); None: off
    ema_params = ema = None
    if ema_decay:
        assert param_dtype == torch.float32, "ema_decay needs fp32 parameters (param_dtype: float32)"
        ema_params = [p for g in groups for p in g["params"]]
        ema = [p.detach().clone() for p in ema_params]
        print(f"ema_decay: {ema_decay} (warm-up {ema_decay_at(0, ema_decay):.2f} -> {ema_decay}); extra memory "
              f"{sum(e.numel() for e in ema) * 4 / 2**30:.1f} GB; checkpoints best_ema/ and final_ema/ are written next to best/ and final/")

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

    def do_eval(step, key="eval"):
        model.eval()
        lg, kept = predict_logits(model, tok, meta, val, cfg.get("eval_batch_size", 16), device, use_autocast)
        model.train()
        r = evaluate(lg, kept)["all"]
        rec = {"step": step, "val_nll": r["nll"], "val_acc": r["acc"], "val_ece": r["ece"], "val_n": r["n"]}
        print(rec)
        log.write(json.dumps({key: rec}) + "\n")
        log.flush()
        return r["nll"]

    model.train()
    step, micro, best, t0, skipped, seen = 0, 0, float("inf"), time.time(), 0, 0
    best_ema = float("inf")
    ep, skip_next = 0, 0
    train_params = [p for g in groups for p in g["params"]]
    snapshot_every, stop_after = cfg.get("snapshot_every"), cfg.get("stop_after_step")
    snap_path = run_dir / "resume" / "state.pt"

    def save_snapshot():
        """Crash-safe state (D37): trainable parameters, optimizer, averaged weights, counters and the data position.
        Written to a temporary file and renamed, so a kill during the write leaves the previous snapshot intact."""
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        st = {"version": 1, "step": step, "micro": micro, "epoch": ep, "batches_seen": batches_seen, "seen": seen,
              "skipped": skipped, "elapsed": time.time() - t0, "best": best, "best_ema": best_ema,
              "total_steps": total_steps, "params": [p.detach().cpu() for p in train_params], "opt": opt.state_dict(),
              "ema": [e.cpu() for e in ema] if ema_decay else None}
        tmp = snap_path.with_suffix(".tmp")
        torch.save(st, tmp)
        os.replace(tmp, snap_path)
        print(f"snapshot written at step {step}: {snap_path} ({snap_path.stat().st_size / 2**30:.1f} GB)")

    batches_seen = 0
    if cfg.get("resume"):
        if snap_path.exists():
            st = torch.load(snap_path, map_location="cpu", weights_only=False)
            assert st["total_steps"] == total_steps and len(st["params"]) == len(train_params), \
                "the snapshot does not belong to this configuration (total steps or parameter list differ)"
            for p, t in zip(train_params, st["params"]):
                p.data.copy_(t.to(p.dtype))
            opt.load_state_dict(st["opt"])
            if ema_decay and st["ema"] is not None:
                for e, t in zip(ema, st["ema"]):
                    e.copy_(t)
            step, micro, seen, skipped = st["step"], st["micro"], st["seen"], st["skipped"]
            best, best_ema, ep, skip_next = st["best"], st["best_ema"], st["epoch"], st["batches_seen"]
            t0 = time.time() - st["elapsed"]
            print(f"resumed from {snap_path}: step {step} of {total_steps}, epoch {ep}, {skip_next} batches skipped")
            del st
        else:
            print(f"resume requested but {snap_path} does not exist: starting from the beginning")

    def eval_and_save(step):
        """Validation of the training weights (and of the averaged weights when ema_decay is set); best checkpoints."""
        nonlocal best, best_ema
        nll = do_eval(step)
        if nll < best:
            best = nll
            save_model(model, tok, run_dir / "best", {**meta, "step": step, "val_nll": nll})
        nll_e = None
        if ema_decay:
            with ema_scope(ema_params, ema):
                nll_e = do_eval(step, "eval_ema")
                if nll_e < best_ema:
                    best_ema = nll_e
                    save_model(model, tok, run_dir / "best_ema", {**meta, "step": step, "val_nll": nll_e, "ema_decay": ema_decay})
        return nll, nll_e

    done = False
    while not done:
        loader = DataLoader(_Pairs(train), batch_sampler=LengthBucketBatches(train, bs, seed, ep, skip=skip_next),
                            collate_fn=collate, num_workers=cfg.get("num_workers", 2),
                            persistent_workers=False)
        batches_seen, skip_next = skip_next, 0
        for batch, decs, dropped in loader:
            batches_seen += 1
            skipped += dropped
            if not decs:
                continue
            seen += len(decs)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_autocast):
                logits = model(batch)
            logits = logits.float() + batch["logit_offset"] if "logit_offset" in batch else logits.float()  # sampled softmax (D33)
            loss, ce, emd = decision_loss(logits, batch, cfg.get("emd_lambda", 0.5))
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
            if ema_decay:
                ema_update(ema, ema_params, ema_decay_at(step, ema_decay))
            if step % cfg.get("log_every", 10) == 0 or step == 1:
                rec = {"step": step, "loss": loss.item(), "ce": ce.mean().item(), "grad_norm": float(gn),
                       "lr": opt.param_groups[0]["lr"], **({"lr_groups": [g["lr"] for g in opt.param_groups]} if dividers else {}),
                       "elapsed_s": round(time.time() - t0, 1), "skipped": skipped,
                       "ex_per_s": round(seen / max(1e-9, time.time() - t0), 2),
                       "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else None}
                print(rec)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            if step % cfg.get("eval_every", 1000) == 0:
                eval_and_save(step)
            if snapshot_every and step % snapshot_every == 0 and step < total_steps:
                save_snapshot()
            if stop_after and step >= stop_after and step < total_steps:   # test and maintenance: stop cleanly after a snapshot
                save_snapshot()
                print(f"stop_after_step reached at step {step}; continue with resume=true")
                return
            if step >= total_steps:
                done = True
                break
        ep += 1
    nll, nll_e = eval_and_save(step)
    save_model(model, tok, run_dir / "final", {**meta, "step": step, "val_nll": nll})
    if ema_decay:
        with ema_scope(ema_params, ema):
            save_model(model, tok, run_dir / "final_ema", {**meta, "step": step, "val_nll": nll_e, "ema_decay": ema_decay})
    print(f"done: steps={step} best_val_nll={best:.4f} skipped_too_long={skipped}"
          + (f" best_val_nll_ema={best_ema:.4f}" if ema_decay else ""))


if __name__ == "__main__":
    main()
