# Mimir-Decide

Mimir-Decide is a research project. It changes the language model DFM Mimir v1.5 into a decision model.

A decision model reads a text and answers questions about it. It does not write text. Each answer is a set of probabilities.

## Example

This example is made up.

| Part | Content |
|---|---|
| State | "Customer writes: my card does not work abroad." |
| Question | "Which team must handle this?" |
| Options | billing, card support, fraud |
| Answer | card support 0.82, fraud 0.12, billing 0.06 |

The model supports 3 types of question:

- **Choice:** Select one option from a list. The answer is one probability for each option.
- **Noul:** Answer a yes-or-no question. The answer is the probability of yes.
- **Score:** Rate the text on an ordered scale. The answer is one probability for each step of the scale.

## Status

The code runs on a GPU server. The first pilot is complete. It used one seed.

The slot model and the letter baseline give the same quality. On validation data they differ by less than 0.005 in accuracy. Mimir already knows most of this data. The result does not show that the model generalizes.

The untrained model with the letter prompt reaches 0.669 accuracy. Fine-tuning raises this to 0.835. A change in the order of the options changes the accuracy by less than 0.5 points. Both models are close to calibrated before the post-hoc step. The next step is a test on tasks with label sets that the training data does not contain. After that, a stronger baseline with a trainable output head follows. The experiment pages in `docs/experiments/` list the steps.

## How it works

Two models use the same data, the same backbone, the same loss and the same calibration. They differ only in how they give the answer.

- **Slot model (main model):** Each option is a marked place in the prompt. The model gives one score for each marked place. A softmax turns the scores into probabilities.
- **Letter baseline:** The options have the letters A, B and C. The model answers with a letter. The project uses this model as a comparison.

After training, the project calibrates the model. It fits one temperature for each question type on validation data. The temperature makes the confidence of the model closer to its accuracy on the validation labels.

Mimir is built on HRM-Text (Hierarchical Reasoning Model for text). HRM-Text is a recurrent model. It has a slow module and a fast module.

## Requirements

- Python 3.11 or later.
- A Python environment. Use a `uv` environment in the project root (`.venv`) or an active conda environment.
- Internet access to download the model and the data from Hugging Face.
- One GPU to train the real model. The plan assumes 40 GB of GPU memory. This value is not measured.

A computer with no GPU can run the tests and the checks with the small model.

## Quick start

1. Clone the repository: `git clone git@github.com:lgalke/mimir-decide.git`
2. Go to the folder: `cd mimir-decide`
3. Set up the environment. Use option A or option B.
   - Option A (uv): run `uv sync --extra dev`. This makes the folder `.venv` in the project root. Then run `source .venv/bin/activate`.
   - Option B (conda): run `conda activate <your-environment>`. Then run `pip install -e ".[dev]"`.
4. Check the GPU: `python -c "import torch; print(torch.cuda.is_available())"`
5. Run the tests: `python -m pytest tests -q -m "not network"`
6. Check the documentation: `python -m mimir_decide.okf --check`

In this README, `python` means the Python of the active environment. Run all commands from the project root. If a `.venv` folder exists and no environment is active, activate the `.venv` or put `uv run` before the command.

The tests take about 1 minute. The 4 tests that need the internet are not in this run. Remove `-m "not network"` to include them.

## Build the data

Do a small build first. It takes about 1 minute:

```bash
python -m mimir_decide.build_mixture --limit 300 --output_dir ~/mimir-decide-data/smoke
```

Do the full build when the small build is correct. The full build needs time, disk space and a network connection:

```bash
python -m mimir_decide.build_mixture --config configs/data.yaml
```

The build writes the data to `~/mimir-decide-data` by default. Change `output_dir` and `cache_dir` in `configs/data.yaml` to use other places.

## Train and evaluate

The script `scripts/run_pilot.sh` does these steps for both models:

1. Build the data.
2. Compare the data with the training data of Mimir.
3. Train the slot model and the letter baseline.
4. Calibrate each model.
5. Evaluate each model on the validation data and on the held-out tasks.

The script does not use the test split. It writes each run to the folder `runs/` in the project. Git tracks only the small result files there.

Use `python -m mimir_decide.compare` to compare two runs side by side. The experiment pages in `docs/experiments/` give the exact commands and the decision rules.

## Folders

| Folder or file | Content |
|---|---|
| `mimir_decide/` | The Python code: data converters, mixture builder, models, training, evaluation. |
| `configs/` | Settings for the data build and for the two models. |
| `scripts/` | The pilot script and a script that makes a small random model for checks. |
| `tests/` | The automatic tests. |
| `docs/` | The knowledge base: research, design decisions, open questions, experiments, observations and a log. |
| `material/` | Reading material. It is not part of the code. |
| `AGENTS.md` | Instructions for AI agents that continue the work. |

## Rules for data safety

**CAUTION: Do not train on the test split. Do not use the test split to make choices.** Use it one time for each finished model. Add `--final` to the command. The program records each use of the test split.

Use only data with a permissive licence. The code removes data with an unknown licence or a non-commercial licence.

The training mix of Mimir v1.5 includes many public data sets. Examples are tasksource and FLAN. Most data that this project uses is in this group. A good result on such data can show only that the model knows the data. It does not prove that the model generalizes. The tools mark these sources by name in each result.

## Documentation

Start with `docs/index.md`. The documentation uses the Open Knowledge Format (OKF). Each page is a Markdown file with a header that has a `type`.

- `docs/design/` has 1 page for each design decision. Each page gives the context, the decision and the consequences.
- `docs/questions/` has 1 page for each open question.
- `docs/experiments/` has the experiment plans. Each page shows its status and its results.
- `docs/observations/` has the findings so far, with the evidence.
- `docs/log.md` is the dated record of all changes.

The command `python -m mimir_decide.okf --write` makes the index pages. The tests fail if an index page is old.

## For AI agents

Read `AGENTS.md` before you change anything. It has the hard rules, the git workflow and the order of the experiments.

## Licence and credits

This repository has no licence file yet. Ask the owner before you reuse the code.

The model and the data have their own licences. DFM Mimir uses the Apache-2.0 licence. The file `docs/design/dataset-selection.md` lists the licence policy and the data sources.

This project uses these works:

- DFM Mimir, from Danish Foundation Models.
- HRM-Text and the Hierarchical Reasoning Model.
- The data sets tasksource, bekko, LocalLLaMA typed-decisions, MASSIVE and HelpSteer2.
- Jev, from TypeSafe AI. Jev is the model that this project follows. The internals of Jev are not public.
