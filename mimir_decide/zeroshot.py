"""Zero-shot reference: the UNTRAINED base model with the letter prompt.

    python -m mimir_decide.zeroshot --config configs/train_baseline.yaml --run_dir $RUNS/letter-zeroshot
    python -m mimir_decide.calibrate --run_dir $RUNS/letter-zeroshot --checkpoint zero-shot --data_dir $DATA
    python -m mimir_decide.evaluate  --run_dir $RUNS/letter-zeroshot --checkpoint zero-shot --data_dir $DATA --split validation

It answers: how much of the fine-tuned letter baseline's quality was already in Mimir v1.5, and how much did
fine-tuning add? The checkpoint stores only the prompt settings and the tokenizer; `load_model` reads the weights
from the hub at the pinned revision. Calibration (3 temperatures) is optional; evaluate with `--no_calibration
--tag uncal` for the raw numbers. The prompt is ours, not Mimir's own training format, so the result is a lower
bound on what Mimir can do zero-shot.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import yaml


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_baseline.yaml", help="letter-baseline config (base model, prompt length)")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--name", default="zero-shot", help="checkpoint folder name inside the run directory")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config))
    assert cfg["model_type"] == "letter", "the zero-shot reference uses the letter prompt (configs/train_baseline.yaml)"

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer") or cfg["base_model"],
                                        revision=cfg.get("tokenizer_revision", cfg.get("revision")))
    run = Path(a.run_dir).expanduser()
    out = run / a.name
    out.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(out / "tokenizer")
    torch.save({}, out / "extra.pt")
    meta = {"model_type": "letter", "max_len": cfg["max_len"], "marker_id": tok.convert_tokens_to_ids("<unused0>"),
            "base_model": cfg["base_model"], "revision": cfg.get("revision"), "zero_shot": True, "step": 0}
    (out / "decision_meta.json").write_text(json.dumps(meta, indent=1))
    shutil.copy(a.config, run / "config.yaml")
    print(f"zero-shot checkpoint written to {out} (weights are loaded from {cfg['base_model']}@{cfg.get('revision')})")


if __name__ == "__main__":
    main()
