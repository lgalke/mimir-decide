"""Name-level audit of our data sources against the Mimir v1.5 (DFM11) training mixture.

Mimir's base data (`data/tokenized_dfm10`) is NOT public, so only dataset NAMES can be compared (via the published
`dfm11_sampling_policy.yaml` prefixes and `training_data_manifest.json`). A name match means "this collection or task
family was in Mimir's post-training mixture (possibly as generative instructions, possibly only some splits)", not that
specific eval rows were seen. Row-level overlap cannot be established from public artifacts.

Output: <data_dir>/reports/mimir_overlap.json (+ .md). evaluate.py reads `sources_seen_by_mimir` and flags those sources.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

REPO = "danish-foundation-models/DFM-Mimir-v1.5"
REVISION = "521b40b36a79918014544b970d4c2669ff1530eb"

# Collection-level rules: our dataset collection -> (policy prefixes that cover it entirely or partly, note)
COLLECTION_RULES = {
    "tasksource": (["tasksource__"], "Mimir's mixture contains the tasksource collection; our typed-decision "
                   "tasksource data is a re-formatting of (largely) the same upstream tasks."),
    "bekko": (["tasksource__", "flan__", "posttrain_natural_instructions__"],
              "bekko subsets are built from public NLP datasets that overlap tasksource/FLAN/Natural-Instructions."),
}
# Task-family keywords that indicate FLAN / Natural-Instructions style coverage (name-level, heuristic)
FAMILY_PREFIXES = ["flan__", "flan_factual__", "flan__cot_", "posttrain_natural_instructions__",
                   "sapient-synth-", "allenai_tulu_3_sft_mixture__", "allenai_tulu_v2_sft_mixture__"]
FAMILY_TASKS = ["mnli", "snli", "anli", "rte", "boolq", "sst", "ag_news", "agnews", "imdb", "qnli", "wnli", "cola",
                "mrpc", "qqp", "squad", "arc", "commonsense_qa", "hellaswag", "winogrande", "piqa", "copa",
                "multirc", "record", "wic", "wsc", "cb", "trec", "yelp", "amazon_polarity", "emotion", "go_emotions",
                "banking77", "clinc", "civil_comments", "toxic", "aqua_rat", "gsm8k", "math", "drop", "quail"]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def task_token(source: str) -> str:
    return norm(source.split("/", 1)[-1])


def audit(sources: dict[str, str], policy_prefixes: list[str]) -> dict:
    """sources: source name -> collection (dataset). Returns report dict."""
    pol_norm = {p: norm(p) for p in policy_prefixes}
    matches: dict[str, list[str]] = {}
    for src, coll in sources.items():
        hit: list[str] = []
        tok = task_token(src)
        for p, pn in pol_norm.items():
            core = re.sub(r"^(dfm\d+|dfm_?\d*)_?", "", pn)
            if core and len(core) >= 4 and (core in tok or (len(tok) >= 4 and tok in core)):
                hit.append(f"name:{p}")
        if coll in COLLECTION_RULES:
            for p in COLLECTION_RULES[coll][0]:
                if p in policy_prefixes:
                    hit.append(f"collection:{p}")
        if any(t == tok or tok.startswith(t + "_") or tok.endswith("_" + t) for t in FAMILY_TASKS):
            hit += [f"family:{p}" for p in FAMILY_PREFIXES if p in policy_prefixes][:2]
        if hit:
            matches[src] = sorted(set(hit))
    # Alias rule: a source whose collection/task name also appears as a path component of a flagged tasksource
    # source (tasksource re-packages e.g. HelpSteer2) is covered by Mimir's `tasksource__` data too.
    ts_parts = {norm(part) for src, coll in sources.items() if coll == "tasksource"
                for part in src.split("/")[1:]}
    for src, coll in sources.items():
        if coll == "tasksource" or src in matches:
            continue
        names = {norm(src.split("/")[0]), norm(coll)}
        if "tasksource__" in policy_prefixes and names & ts_parts:
            matches[src] = [f"alias:tasksource contains '{sorted(names & ts_parts)[0]}'"]
    return {"sources_seen_by_mimir": sorted(matches), "matches": matches,
            "n_sources_checked": len(sources), "n_flagged": len(matches)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    a = ap.parse_args(argv)
    data = Path(a.data_dir).expanduser()
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    pol = yaml.safe_load(open(hf_hub_download(REPO, "dfm11_sampling_policy.yaml", revision=REVISION)))
    prefixes = [e["prefix"] for e in pol]
    manifest = json.loads(Path(hf_hub_download(REPO, "training_data_manifest.json", revision=REVISION)).read_text())
    mm = json.loads((data / "mixture_manifest.json").read_text())
    sources = {}
    for f in ["train.parquet", "validation.parquet", "calib.parquet", "heldout_tasks/heldout.parquet",
              "eval/test/test.parquet", "extra_eval/extra_eval.parquet"]:
        p = data / f
        if p.exists():
            t = pq.read_table(p, columns=["source", "dataset"]).to_pylist()
            sources.update({r["source"]: r["dataset"] for r in t})
    rep = audit(sources, prefixes)
    rep.update({
        "mimir_repo": REPO, "mimir_revision": REVISION, "policy_prefixes": len(prefixes),
        "public_addition_packages": [p["repo_id"] for p in manifest["public_addition_packages"]],
        "collection_notes": {k: v[1] for k, v in COLLECTION_RULES.items()},
        "mixture_manifest_counts": mm["counts"],
        "limitations": [
            "Name-level only; DFM10 base data (tokenized_dfm10) is not public.",
            "A flag means the collection/task family was in Mimir's mixture, not that specific eval rows were seen.",
            "Absence of a flag is NOT proof of no overlap (name heuristics can miss renamed datasets).",
            "Row-level (text hash) overlap was not computed.",
        ],
    })
    out = data / "reports"
    out.mkdir(exist_ok=True)
    (out / "mimir_overlap.json").write_text(json.dumps(rep, indent=1))
    lines = ["# Mimir v1.5 overlap audit (name-level)", "",
             f"Checked {rep['n_sources_checked']} sources against {rep['policy_prefixes']} policy prefixes; "
             f"flagged {rep['n_flagged']}.", "", "## Limitations", *[f"- {x}" for x in rep["limitations"]], "",
             "## Flagged sources", *[f"- `{s}`: {', '.join(m)}" for s, m in sorted(rep["matches"].items())]]
    (out / "mimir_overlap.md").write_text("\n".join(lines) + "\n")
    by_coll: dict[str, list[int]] = {}
    for s, c in sources.items():
        by_coll.setdefault(c, [0, 0])
        by_coll[c][0] += 1
        by_coll[c][1] += s in rep["matches"]
    print(json.dumps({"flagged_by_collection(sources, flagged)": by_coll}, indent=1))


if __name__ == "__main__":
    main()
