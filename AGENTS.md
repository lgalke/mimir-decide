# AGENTS.md — Mimir-Decide handoff

You are continuing a project that turns **DFM Mimir v1.5** (an open 1B-class HRM-Text language model) into a **Jev-like decision model**: text state + typed questions in, calibrated probabilities out, no generated text. Work so far was done on a Mac without a GPU. **You are on the GPU cluster: the real model has never been run yet.** Your first job is to run the planned experiments in order and record what happens.

Read this file fully, then `docs/index.md`. Everything the previous agent knew that is not in the code is in `docs/`.

## 1. State of the project (2026-10-04)

Done and tested (53 tests: 50 offline, 3 need network):
- Dataset converters for 5 sources, mixture builder with leakage guards and licence policy, Mimir overlap audit, slot-readout model and letter-logit baseline, training loop, calibration, evaluation, comparison, latency benchmark.
- Documentation as an OKF bundle in `docs/`: 22 decision records, 13 open questions, 9 prefilled experiments (all **planned, none run**), 13 observations, update log.

Verified only on a **tiny random HRM** (real tokenizer, CPU) plus unit tests: build → train → calibrate → evaluate works end to end for both models; gradient checkpointing, `num_workers=2` and the `L_bp_cycles` override run.

**Not verified (you will hit these first):**
- Loading and training the real 1.8B checkpoint; GPU memory, speed, whether `[3,3]` backprop fits ([Q03](docs/questions/q03-real-model-memory-and-speed.md)).
- CUDA-specific paths: `torch.autocast` bf16, `fused=True` AdamW, gradient checkpointing on the real model, `sdpa` with the PrefixLM mask.
- The **full** data build (only `--limit 4000` trial builds were run; takes network, disk and time).
- Any statement about decision quality. No real-model number exists. The metrics in smoke runs are meaningless (random model).

## 2. Set up (do this first)

**What you received:** a git repository. Branch `main` at commit `ad89493` ("first commit") is the handoff baseline: all code, configs and `docs/`, 121 tracked files, no data or checkpoints. Its remote is `origin` = `git@github.com:lgalke/mimir-decide.git`, and `main` tracked `origin/main` with nothing ahead or behind when this was written (verify with `git fetch && git status -sb`). Nothing from the previous machine's `~/mimir-decide-data` (smoke mixtures, tiny model, runs) exists here: rebuild what you need. **Read section 9 (git workflow) before your first commit.**

```bash
git clone git@github.com:lgalke/mimir-decide.git && cd mimir-decide   # or cd into the existing checkout
git switch -c cluster/<short-topic>                                    # never work directly on main (section 9)
# Environment: use ONE option. Do not create a second environment if one exists already.
uv sync --extra dev && source .venv/bin/activate    # option A: uv environment in the project root (.venv)
# conda activate <env> && pip install -e ".[dev]"   # option B: an activated conda environment
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m pytest tests -q -m "not network"          # ~1 min; add network tests if the cluster has internet
python -m mimir_decide.okf --check                  # docs must be valid before you start
```

In this file, `python` means the Python of the active environment. Run all commands from the repository root. If a `.venv` exists in the root and no environment is active, activate it or prefix commands with `uv run` ([D23](docs/design/d23-active-python-environment.md)).

- `uv.lock` is tracked and was resolved on macOS. If torch on the cluster is not a CUDA build, install the right wheel in your environment (uv `.venv` or conda) and **record it** in `docs/log.md`; do not commit a cluster-specific `uv.lock` change without the owner's agreement (section 9).
- If the clone fails for lack of SSH access from the cluster, ask the owner for a deploy key or another way in. Never put tokens or keys in files, commits or logs.
- Data, caches, runs default to `~/mimir-decide-data` (override: `output_dir`, `cache_dir` in `configs/data.yaml`; `DATA`, `RUNS` for `scripts/run_pilot.sh`; `--set run_dir=...`). Put them on fast scratch, never in a synced folder.
- Hugging Face access: all sources used are public. Pinned revisions are in `configs/*.yaml` (Mimir v1.5 `521b40b36a79918014544b970d4c2669ff1530eb`). If the compute nodes have no internet, pre-download on a login node (`huggingface_hub`, `datasets`) and set `HF_HOME` / `HF_HUB_OFFLINE=1`. The MASSIVE archive is fetched with `urllib` from S3 into `cache_dir`.
- `~/.cache/huggingface` may be large and shared with other projects: do not clear it.

## 3. What to do, in order

Each step has a page in `docs/experiments/` with exact commands, metrics and a **decision rule written before running**. Follow the page; do not improvise the rule afterwards.

1. **[E01 full mixture build](docs/experiments/e01-full-mixture-build.md)** — then read `reports/*` and hand-inspect `ngram_leak_examples.jsonl`. Check `heldout_patterns_unmatched` is empty.
2. **[E02 smoke run on the real model](docs/experiments/e02-smoke-run-real-model.md)** — 50 steps; find a config that fits; record `peak_mem_gb`, `ex_per_s`. Update D04/D09/D10 if needed.
3. **[E03 slot vs letter pilot](docs/experiments/e03-slot-vs-letter-pilot.md)** (`scripts/run_pilot.sh`), ≥2 seeds. This is the central question ([Q04](docs/questions/q04-slot-vs-letter-on-real-model.md)).
4. E04 calibration, E05 option-order robustness, E07 seen-vs-unflagged analysis (from the E03 runs); E06 `L_bp_cycles`; E08 latency. E09 is exploratory and needs code.

After each experiment: fill the results table in its page, write an observation page (copy the template), set the experiment `status`, add a line to `docs/log.md`, regenerate indexes (section 5).

## 4. Hard rules (do not break; ask the project owner before changing any of them)

1. **Never train on, tune on, or casually look at the test split.** `train.py` reads only `train.parquet` and `validation.parquet` and refuses `eval/`, `heldout_tasks/`, `test*`, `heldout*`. `evaluate.py --split test` needs `--final`, is logged in `eval/test_access_log.jsonl`, and should be run at most once per finished model. Choose models and hyperparameters on validation (and held-out tasks), never test. ([D11](docs/design/d11-split-isolation-and-test-access.md))
2. **Do not weaken the leakage guards** in `build_mixture.py` (exact state hash, group id, cross-source n-gram fingerprints, final assertions). If a guard fires unexpectedly, investigate with `reports/ngram_leak_examples.jsonl`; do not disable it. Each guard has a test that fails when it is removed. ([D12](docs/design/d12-ngram-fingerprint-guard-cross-source-only.md))
3. **Permissive licences only** ([D02](docs/design/d02-permissive-licences-only.md)); bekko "qualified" subsets stay excluded ([D13](docs/design/d13-bekko-licence-and-test-handling.md)). Do not add data with unknown or non-commercial licences.
4. **Held-out tasks are fixed before any training** ([D15](docs/design/d15-held-out-task-massive-nb-no.md)). It is currently provisional: `massive/nb-NO` is probably *not* clean (Mimir saw `tasksource__`). **Before the first real training run, raise this with the owner** (see Q01); do not change `held_out_sources` after a model has been trained.
5. **Mimir overlap is a standing caveat.** Nearly all public sources are flagged "seen by Mimir" (name-level). Every reported result must say which sources are flagged; use `python -m mimir_decide.compare` (seen / not-flagged blocks). Do not claim generalisation from flagged sources.
6. Do not describe outputs as "probability of correctness" ([Q07](docs/questions/q07-soft-label-semantics-vs-correctness.md)); calibration is to the given labels.
7. Report faithfully: failing tests, OOMs, surprising or negative results go into observations and the log, with numbers and run paths.

**Ask the owner before:** changing the held-out choice, licence policy, base model revision, or the decision rule of an experiment after seeing results; adding data sources; running `--split test --final`; pushing to `origin` or publishing anything (branches, models, datasets, docs; see section 9); anything that downloads or stores very large data outside the paths above.

## 5. How to work on docs (OKF) — required for every change

`docs/` is an [Open Knowledge Format](https://okf.md/spec) bundle: markdown pages with YAML frontmatter, a non-empty `type`, bundle-absolute links (`/design/d05-....md`), generated `index.md` files, and a reserved `log.md`.

```
docs/research/      what is known (Jev, HRM-Text, Mimir, sources)
docs/design/        decision records D01..: Context / Decision / Consequences / Revisit when; plus how-to (training-setup)
docs/questions/     Q01..: open questions with how to resolve them
docs/experiments/   E01..: plans with decision rules; results go into the table at the bottom
docs/observations/  O01..: dated evidence, with how to reproduce
docs/log.md         newest-first dated log; one bullet per decision, finding, fix or verification
```

Workflow:
1. Copy the right `template-*.md`, use the next id, keep frontmatter valid YAML (**quote values containing `: `**).
2. A behaviour change needs a decision page (or an update of the existing one; never silently change an accepted decision — mark it `superseded` and link both ways) **and** a log line.
3. Answered question: set `status: answered`, add an **Answer** section citing an observation.
4. Regenerate and check: `python -m mimir_decide.okf --write && python -m mimir_decide.okf --check` (also enforced by `tests/test_okf_bundle.py`).
5. Run the tests before finishing.

## 6. Repository map

```
AGENTS.md                   this file
README.md                   overview for humans, written in ASD-STE100 style (keep that style when you edit it)
configs/data.yaml           sources, caps, held-out tasks, pinned dataset revisions
configs/train_slot.yaml     slot readout (main model)       configs/train_baseline.yaml   letter-logit baseline
mimir_decide/
  schema.py licenses.py     Decision record, hashes, fingerprints; licence policy
  convert/*.py              tasksource, bekko, localllama, massive, helpsteer2 -> Decision
  build_mixture.py          eval pass -> train pass with guards -> caps -> parquet + manifest + reports
  audit_mimir_overlap.py    name-level overlap with Mimir's training policy
  formatting.py model.py    prompt rendering/collation; SlotDecisionModel, LetterBaseline, loss, save/load
  train.py calibrate.py evaluate.py inference.py metrics.py
  compare.py bench.py okf.py
scripts/run_pilot.sh        build -> audit -> train both -> calibrate -> evaluate (validation + heldout)
scripts/make_tiny_model.py  random tiny HRM with the real vocabulary (smoke tests without the big model)
tests/                      53 tests; marker `network` needs internet
docs/                       the OKF knowledge base (section 5)
```

Run directories contain `config.yaml`, `resolved_config.json`, `mixture_manifest.json`, `train_log.jsonl`, `best/`, `final/`, `calibration.json`, `eval/*.json`. Checkpoints are bf16 inference weights (`lm/`) + `extra.pt` (slot head/marker) + tokenizer + `decision_meta.json`.

Useful commands:
```bash
python -m mimir_decide.train --config configs/train_slot.yaml --set data_dir=$DATA run_dir=$RUNS/x max_steps=50   # overrides are YAML; quote lists: 'L_bp_cycles=[0,3]'
python -m mimir_decide.calibrate --run_dir $RUNS/x --data_dir $DATA
python -m mimir_decide.evaluate  --run_dir $RUNS/x --data_dir $DATA --split validation|heldout [--order_seed 1] [--no_calibration]
python -m mimir_decide.compare   $RUNS/a $RUNS/b --file validation.json
python -m mimir_decide.bench     --run_dir $RUNS/x --data_dir $DATA
```
Out-of-memory ladder for E02: `batch_size=4 grad_accum=8` → `max_len=1024` → `L_bp_cycles=[0,3]` → `param_dtype=bfloat16`. Log which rung worked.

## 7. Gotchas learned the hard way

- Dataset cards were wrong or incomplete several times ([O03](docs/observations/o03-dataset-cards-differ-from-the-data.md)): verify against real rows and watch the `drops` counters in `mixture_manifest.json`. A suspiciously large drop count is a bug until proven otherwise.
- Do not run two builds into the same output dir; they share `cache_dir` (downloads are atomic now) but write the same outputs.
- Smoke numbers from the tiny model mean nothing ([O12](docs/observations/o12-uncalibrated-tiny-model-smoke-numbers.md)). `scripts/make_tiny_model.py` + `--set base_model=<tiny> revision=null tokenizer=danish-foundation-models/DFM-Mimir-v1.5 tokenizer_revision=<rev> device=cpu` is the offline check.
- YAML 1.1 reads `1e-3` as a string (the loader converts overrides, but write `1.0e-3` in files).
- `num_hidden_layers` in the HRM config means layers *per stack*; the config inflates it internally. FlashAttention is rejected with `prefix_lm`; use `sdpa`.
- tasksource is large and read by streaming; bekko files come through the HF cache. Use `--limit N` to smoke-test any change to converters or the builder.
- Shell habits of the previous machine (macOS/zsh) do not apply on the cluster, but on any shell: pass overrides as separate arguments, and avoid interactive prompts in scripts.

## 8. Finishing a session

Leave the repo in a state the next agent can use: tests green, `okf --check` clean, run directories named in the experiment tables, and a final `docs/log.md` entry (under today's date) listing what was run, what the numbers were, what failed, and the recommended next step. If you could not run something, say so there rather than omitting it. Then **commit everything on your branch** (working tree clean, `git status` shows nothing) and put the branch name and last commit hash in that final log entry. Push only if the owner has allowed it (section 9).

## 9. Git workflow

Baseline: `main` @ `ad89493`. The owner reviews and merges; you do not.

- **Branch.** Work on `cluster/<short-topic>` (for example `cluster/e02-real-model-smoke`), created from an up-to-date `main`. Never commit directly on `main`.
- **Commit often, in coherent steps.** One commit per logical change that includes its tests and its docs (decision page or observation, log line, regenerated indexes). Record each finished experiment in its own commit. Commit messages: imperative subject of at most 72 characters, a body that says *why* and quotes the key numbers, and the experiment or decision id (`E02`, `D09`). Use `git commit -m ...`; do not rely on an editor.
- **Before every commit:** `python -m pytest tests -q -m "not network"`, `python -m mimir_decide.okf --check`, and `git status` to confirm that only intended files are staged.
- **Never commit** data, mixtures, caches, checkpoints, logs, virtual environments, Hugging Face tokens, SSH keys, or anything larger than about 5 MB. **One exception, decided by the owner on 2026-10-09:** the small evaluation result files `runs/<run>/eval/*.json` (about 100 KB each) are tracked, so results travel with the repository. `.gitignore` allows exactly these files inside `runs/` and ignores the rest (checkpoints in `best/` and `final/`, `train_log.jsonl`, `calibration.json`, `eval/test_access_log.jsonl`). It also ignores `.venv/`, `data/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.DS_Store`, `docs/.obsidian/` and vim swap files. Check the size of every file before you commit it. Mixtures and the other run files live under `~/mimir-decide-data`; the default `RUNS` is outside the repository, so copy the eval files into `runs/<run>/eval/` yourself if you want to track them. Put the key numbers into the experiment table and an observation page as well.
- **Pushing needs the owner's permission.** Local commits are expected and need no permission. Ask once at the start whether you may push your own `cluster/*` branch to `origin`; until told yes, do not push. Even when allowed, never push to `main`, force-push, rewrite published history, delete branches, create tags or releases, or open pull requests unless asked.
- **Generated and shared files.** On a merge or rebase conflict in any `index.md`, do not hand-merge: resolve the page files, then run `python -m mimir_decide.okf --write`. In `docs/log.md` keep both sides' bullets, newest date first. IDs (`D`, `Q`, `E`, `O`) must stay unique and sequential (a test checks this): if two branches took the same next id, the later one renumbers.
- **Dependencies.** If you change `pyproject.toml`, run `uv lock` and commit both files together with a log line. A lock change that only exists to suit this cluster (for example a CUDA-specific torch pin) is not committed unless the owner agrees.
- **Undoing things.** Prefer `git revert` or a new commit over rewriting history. Do not use `git reset --hard`, `git clean -fdx` or `git checkout -- .` on a tree with uncommitted work you have not inspected.
- The process decision is recorded in [D22](docs/design/d22-git-workflow.md); if the owner changes the rules, update that page and this section together.
