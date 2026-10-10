# Design

Decision records (one page per decision, with context and consequences) and how-to references.

## Decision

* [D01: Base model is Mimir v1.5](/design/d01-base-model-mimir-v1-5.md) - [accepted] Start from DFM Mimir v1.5 (pinned revision), not v1.
* [D02: Permissive licences only](/design/d02-permissive-licences-only.md) - [accepted] Only train and evaluate on data whose licences are on a permissive allowlist; unknown or non-commercial is excluded.
* [D03: Skip in-house data for now](/design/d03-skip-in-house-data-for-now.md) - [accepted] DaLA, GEC, Arena, audit and tool examples are not converted yet.
* [D04: Single GPU node, plain PyTorch loop](/design/d04-single-gpu-plain-loop.md) - [accepted] No SLURM, no multi-GPU; a simple single-process training loop.
* [D05: One shared slot readout for Noul, Choice and Score](/design/d05-slot-readout-heads.md) - [accepted] Options are marked slots in the prompt; one scalar logit per option read from the final H-state; no per-task heads.
* [D06: Letter-logit baseline](/design/d06-letter-logit-baseline.md) - [accepted] Compare against reading A/B/C letter logits from the unchanged language-model head.
* [D07: Option marker is a learned vector, not a new token](/design/d07-learned-marker-vector.md) - [accepted] Placeholder positions in the prompt get a learned embedding instead of an added vocabulary token.
* [D08: Per-kind logit scale, no per-kind bias](/design/d08-per-kind-scale-not-bias.md) - [accepted] A learned scale per kind (noul, choice, score); a bias would be invisible to the softmax.
* [D09: Keep the checkpoint's L_bp_cycles [3,3]](/design/d09-keep-checkpoint-l-bp-cycles.md) - [accepted] Fine-tune with the same gradient routing the checkpoint was trained with.
* [D10: fp32 master weights with bf16 autocast](/design/d10-fp32-master-weights-bf16-autocast.md) - [accepted] Parameters in float32, forward in bf16 on CUDA.
* [D11: Split isolation and test-access protocol](/design/d11-split-isolation-and-test-access.md) - [accepted] Train on train splits only; eval data of every source is hashed first; test is touched only with --final and logged.
* [D12: N-gram fingerprint guard compares against other sources only](/design/d12-ngram-fingerprint-guard-cross-source-only.md) - [accepted] Rendering-independent content check, applied across sources but not within one.
* [D13: bekko: strict licence status, hash-only test](/design/d13-bekko-licence-and-test-handling.md) - [accepted] Use only 'verified' upstream licences; hash bekko test rows but never write them.
* [D14: Carving validation/test where upstream has none](/design/d14-split-carving-where-upstream-has-no-clean-split.md) - [accepted] Deterministic group-wise carve-outs by hashed id.
* [D15: Held-out task: massive/nb-NO (provisional)](/design/d15-held-out-task-massive-nb-no.md) - [accepted] One MASSIVE locale is removed from training and evaluated separately.
* [D16: Exclude ChaosNLI](/design/d16-exclude-chaosnli.md) - [accepted] Not used, not even for evaluation.
* [D17: Sequence and option budget](/design/d17-sequence-and-option-budget.md) - [accepted] max_len 2048, middle truncation of the state, at most 26 options, long states dropped.
* [D18: Loss: soft cross-entropy plus EMD for scores](/design/d18-loss-soft-ce-plus-emd.md) - [accepted] Cross-entropy against the target distribution for all kinds; squared-CDF penalty (weight 0.5) for Score.
* [D19: Calibration: one temperature per kind on held-out validation](/design/d19-calibration-protocol.md) - [accepted] Fit temperatures by LBFGS on soft NLL using half of validation, split by group.
* [D20: Environment and data live outside OneDrive](/design/d20-environment-outside-onedrive.md) - [accepted] Data, caches and runs live in ~/mimir-decide-data, outside the synced folder. The venv-location part is superseded by D23.
* [D21: Documentation is an OKF bundle with generated indexes](/design/d21-docs-as-okf-bundle.md) - [accepted] docs/ follows the Open Knowledge Format; indexes are generated and checked; every decision, finding and result is logged.
* [D22: Git workflow for agents](/design/d22-git-workflow.md) - [proposed] Agents work on cluster/* branches with frequent commits; the owner merges; pushing needs permission; data and checkpoints never enter the repository.
* [D23: Use the active Python environment](/design/d23-active-python-environment.md) - [accepted] Documents and scripts call python from the active environment, a uv .venv in the project root or an activated conda environment. No fixed venv path.
* [D24: Run directories live in the repository](/design/d24-run-directories-in-the-repository.md) - [accepted] Runs default to runs/ in the repository, so the tracked result files (eval JSON, training log) are written where git sees them. Checkpoints stay ignored by git.
* [D25: Leave whole tasks out to test unseen label sets](/design/d25-leave-tasks-out-for-unseen-label-sets.md) - [superseded] E12 trains on a mixture derived from the pilot mixture without ten selected sources, chosen by a measurable label-novelty rule and pre-registered. The pilot configuration stays unchanged.
* [D26: Evaluate unseen label sets on additional tasks, without retraining](/design/d26-evaluate-unseen-label-sets-on-additional-tasks.md) - [proposed] Test transfer to unseen label sets by evaluating the existing models on extra tasks that are not in the training mixture, instead of retraining on a reduced mixture.
* [Dataset selection, licence policy and leakage controls](/design/dataset-selection.md) - [accepted] Which sources feed Mimir-Decide, how each is split, what is excluded and why, and how train/eval separation is enforced.
* [Output heads strategy and the LM-letter baseline](/design/heads-and-baseline.md) - [accepted] One shared open-vocabulary slot readout for Noul/Choice/Score instead of per-task heads, plus the letter-logit baseline to compare against.
* [Mimir to Decision Model, continual training plan](/design/mimir-to-decision-model-plan.md) - [superseded] Proposed design for converting Mimir v1 into a Jev-like decision model. Our synthesis, untested.

## Reference

* [Training setup and how to run it](/design/training-setup.md) - Code layout, commands, configuration knobs, split safeguards and verified HRM-Text facts for the Mimir-Decide scripts.

## Template

* [Template: decision record](/design/template-decision.md) - Copy to design/dNN-slug.md to record a design decision.
