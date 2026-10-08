---
type: Reference
title: Training setup and how to run it
description: Code layout, commands, configuration knobs, split safeguards and verified HRM-Text facts for the Mimir-Decide scripts.
tags: [training, scripts, reproducibility, safeguards]
timestamp: 2026-10-04T00:00:00Z
---

# Training setup

Code lives in the project root next to this bundle (`mimir_decide/`, `configs/`, `scripts/`, `tests/`). Large data, caches and checkpoints go to `~/mimir-decide-data`, deliberately outside OneDrive. The Python environment is the active one ([D23](/design/d23-active-python-environment.md)); commands below use `python` from it.

## Run order

```
uv sync --extra dev                       # uv: creates .venv in the project root; or, in an active conda env: pip install -e ".[dev]"
python -m mimir_decide.build_mixture --config configs/data.yaml [--limit N]
python -m mimir_decide.audit_mimir_overlap --data_dir <mixture>
python -m mimir_decide.train      --config configs/train_slot.yaml      # slot model
python -m mimir_decide.train      --config configs/train_baseline.yaml  # letter baseline
python -m mimir_decide.calibrate  --run_dir <run> --data_dir <mixture>
python -m mimir_decide.evaluate   --run_dir <run> --data_dir <mixture> --split validation|heldout
scripts/run_pilot.sh                      # all of the above for both models
```

Tools added for the planned experiments ([experiments](/experiments/index.md)):

```
python -m mimir_decide.evaluate --run_dir <run> --data_dir <mixture> --split validation --order_seed 1   # option-order robustness
python -m mimir_decide.compare <run_a> <run_b> --file validation.json                                    # side by side, seen vs not flagged
python -m mimir_decide.bench   --run_dir <run> --data_dir <mixture>                                      # latency / throughput
python -m mimir_decide.zeroshot --config configs/train_baseline.yaml --run_dir <run>                        # untrained base model, letter prompt (E10); use --checkpoint zero-shot below
python -m mimir_decide.okf --check | --write                                                              # docs indexes
```

`calibrate` and `evaluate` accept `--limit N` (seeded random sample, smoke tests only); `evaluate` accepts `--tag NAME` to keep raw and calibrated results in separate files.

`train_log.jsonl` records `peak_mem_gb` (CUDA) and `ex_per_s`.

Overrides: `train --set key=value ...` (values are YAML; `1e-3` is also accepted as a float).

## Split safeguards

1. **Converters read each dataset's own splits.** Upstream split labels are normalised (`dev` is `validation`); a row whose split column disagrees with the split it came from is dropped and counted.
2. **Builder order.** The evaluation splits of every source are read first and hashed (exact state hash, group id, and rendering-independent word n-gram fingerprints). The train pass then drops any row that matches. Boilerplate n-grams (present in more than 3 eval states) are ignored so that templated sources are not erased.
3. **Final assertions** on the written files: no train state, group or id also appears in any eval file; only train-split records, no held-out source and only permissive licences in `train.parquet`. The build fails otherwise.
4. **Test access.** `train.py` reads only `train.parquet` and `validation.parquet`, and refuses paths containing `eval/` or `heldout_tasks/` or file names starting with `test`/`heldout`. `evaluate.py --split test` exits unless `--final` is given and then appends to `eval/test_access_log.jsonl` in the run directory.
5. **Calibration data** is half of validation (split by group), never train or test.
6. **Tests.** The three leakage guards were each disabled in turn and the suite failed every time (mutation check); see the [log](/log.md).

## Configuration knobs that matter

| Knob | Default | Note |
|---|---|---|
| `max_len` | 2048 | State is cut from the middle; question and options are never cut; rows where those alone do not fit are skipped and counted |
| `per_source_cap`, `max_total_train`, `sampling_alpha` | 2000, 1M, 0.5 | Temperature-based allocation across sources |
| `L_bp_cycles` | checkpoint's `[3,3]` | `[0,3]` saves memory and compute but changes the training regime |
| `freeze_input_embeddings` | true | 262k x 1536 embedding matrix stays fixed |
| `gradient_checkpointing` | true | |
| `param_dtype` | float32 | fp32 master weights with bf16 autocast on CUDA; bf16 weights are possible for smoke runs |
| `lr_backbone`, `lr_head` | 1e-5, 1e-4 | v1.5 finished pretraining at 1e-5 |

Rough memory with the defaults: fp32 weights (~7 GB) + gradients and AdamW state for about 1B trainable parameters (~12 GB) + activations. A 40 GB GPU should be enough with gradient checkpointing; this has not been measured on a GPU.

## Verified facts about the model code

- `HrmTextModel.forward` accepts `inputs_embeds` and multiplies them by `embedding_scale` itself. The learned marker vector therefore lives in unscaled embedding space.
- `token_type_ids == 1` marks the bidirectional block; right-padding gets 0.
- `num_hidden_layers` in the config means layers per stack; the config inflates it internally.
- The Gemma tokenizer's chat markup tokenizes identically whether written as text or applied through the template, and `A`–`Z` are single tokens.

## Not done yet

No GPU was available in the development environment: the real 1.8B model has not been trained or even loaded in these scripts. Pipeline correctness was checked with a tiny random HRM (same tokenizer and vocabulary) on CPU, plus unit tests. See [open questions](/research/open-questions.md).
