"""hotchpotch/bekko-system-one-dataset-v0 -> Decision.

Rules taken from the dataset card (verified 2026-10-04):
- Active membership is defined ONLY by the root `training-manifest.json` ("train" and "evaluation" lists; the
  evaluation list has split=validation|test). Configs in `quarantine-manifest.json` must not be used.
- Soft targets and candidate order must be preserved.
- Licences are per upstream dataset (`sources.json` -> datasets[name][role].upstream_datasets[*].license).
  A subset is allowed only if ALL its upstream licences pass the permissive policy and every review status is
  "verified" (unless `accept_qualified` is set). Subsets without upstream licence info are excluded.

Row layout: input.state_json (JSON), input.decisions[{id, kind, type, instructions_json, criteria[{id,
description_json, value}], documents[{id, content_json}], scoring}], targets[{decision_id, ids, probabilities}].
Supported decisions: choice / noul / score (via criteria) and ranking-over-documents (as a Choice over the
documents). Everything else is dropped and counted.
"""
from __future__ import annotations

import json
from typing import Iterator, Optional

from .. import licenses
from .common import Counter0, jloads, make_decision, norm_split, render_state

REPO = "hotchpotch/bekko-system-one-dataset-v0"
MAX_DOC_CHARS = 600


def _load_json(name: str, revision: Optional[str]):
    from huggingface_hub import hf_hub_download

    return json.load(open(hf_hub_download(REPO, name, repo_type="dataset", revision=revision)))


def load_manifests(revision: Optional[str] = None) -> dict:
    return {
        "training": _load_json("training-manifest.json", revision),
        "quarantine": _load_json("quarantine-manifest.json", revision),
        "sources": _load_json("sources.json", revision)["datasets"],
    }


def subset_license(name: str, sources: dict, accept_qualified: bool = False) -> tuple[bool, str, str]:
    """Return (allowed, reason, license_string) for a bekko subset."""
    entry = sources.get(name)
    if not entry:
        return False, "no_source_entry", ""
    ups = []
    for role, node in entry.items():
        if isinstance(node, dict):
            ups += node.get("upstream_datasets") or []
    if not ups:
        return False, "no_upstream_license_info", ""
    lic_parts, reasons = [], []
    for u in ups:
        raw = u.get("license")
        lic = ", ".join(raw) if isinstance(raw, list) else str(raw or "")
        ok, why = licenses.check(lic)
        status = u.get("license_review_status")
        if ok and status != "verified" and not accept_qualified:
            ok, why = False, f"license_{status or 'unreviewed'}"
        lic_parts.append(lic)
        if not ok:
            reasons.append(why)
    if reasons:
        return False, reasons[0], "; ".join(sorted(set(lic_parts)))
    return True, "ok", "; ".join(sorted(set(lic_parts)))


def manifest_entries(manifests: dict, split: str) -> list[dict]:
    q = set()
    for k in ("train", "evaluation", "unlabeled"):
        for e in manifests["quarantine"].get(k, []):
            q.add(e["dataset"] if isinstance(e, dict) else e)
    if split == "train":
        entries = manifests["training"]["train"]
    else:
        entries = [e for e in manifests["training"]["evaluation"] if e["split"] == split]
    return [e for e in entries if e["dataset"] not in q]


def _txt(x) -> str:
    v = jloads(x, "")
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def row_to_decisions(r: dict, dataset: str, license_str: str, split: str, drops: Counter0) -> list:
    inp, targets = r["input"], {t["decision_id"]: t for t in r["targets"]}
    state_obj = jloads(inp["state_json"], "")
    out = []
    for dec in inp["decisions"]:
        t = targets.get(dec["id"])
        if t is None:
            drops.inc("bekko_no_target")
            continue
        ids, probs = list(t["ids"]), list(t["probabilities"])
        question = _txt(dec.get("instructions_json"))
        crit, docs = dec.get("criteria") or [], dec.get("documents") or []
        state = render_state(state_obj)
        values = None
        if crit:
            by_id = {c["id"]: c for c in crit}
            if any(i not in by_id for i in ids):
                drops.inc("bekko_target_id_not_in_criteria")
                continue
            kind = dec.get("type")
            if kind not in ("choice", "noul", "score"):
                drops.inc(f"bekko_unsupported_type:{kind}")
                continue
            order = list(range(len(ids)))
            if kind == "score":
                vals = [by_id[i].get("value") for i in ids]
                if any(v is None for v in vals):
                    drops.inc("bekko_score_without_values")
                    continue
                order = sorted(order, key=lambda j: vals[j])
                values = [float(vals[j]) for j in order]
            if kind == "noul":
                # authored definitions of true/false; present as two options, order as authored
                pass
            opts = [_txt(by_id[ids[j]]["description_json"]) for j in order]
            probs = [probs[j] for j in order]
        elif docs and dec.get("kind") == "ranking":
            kind = "choice"
            by_id = {d["id"]: d for d in docs}
            if any(i not in by_id for i in ids):
                drops.inc("bekko_target_id_not_in_documents")
                continue
            opts = [_txt(by_id[i]["content_json"])[:MAX_DOC_CHARS] for i in ids]
        else:
            drops.inc(f"bekko_unsupported_decision:{dec.get('kind')}")
            continue
        d = make_decision(
            id=f"bekko:{r['case_id']}:{dec['id']}", case_id=r["case_id"], group_id=r["group_id"],
            dataset="bekko", source=f"bekko/{dataset}", split=split, language=r.get("language") or "und",
            license=license_str, kind=kind, state=state, question=question, options=opts, target=probs,
            option_values=values, variant=dec.get("kind"),
        )
        if d:
            out.append(d)
        else:
            drops.inc("bekko_empty_target")
    return out


def iter_records(split: str, drops: Counter0, limit: Optional[int] = None, revision: Optional[str] = None,
                 accept_qualified: bool = False, manifests: Optional[dict] = None,
                 excluded: Optional[Counter0] = None) -> Iterator:
    """Yield decisions for `split`. License-excluded subsets are counted in `excluded` (by reason)
    but their rows are NOT downloaded for train; for eval splits they are still yielded with license
    '(excluded)' so the leakage guard can hash them (the builder never writes them)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    m = manifests or load_manifests(revision)
    n = 0
    for e in manifest_entries(m, split):
        name = e["dataset"]
        ok, why, lic = subset_license(name, m["sources"], accept_qualified)
        if not ok:
            if excluded is not None:
                excluded.inc(f"{name}|{why}", e.get("decisions", 0))
            if split == "train":
                continue
        for f in e["data_files"]:
            path = hf_hub_download(REPO, f, repo_type="dataset", revision=revision)
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=512):
                for row in batch.to_pylist():
                    if norm_split(row.get("split")) not in (None, split):
                        drops.inc("bekko_split_column_mismatch")
                        continue
                    for d in row_to_decisions(row, name, lic if ok else f"(excluded:{why})", split, drops):
                        yield d
                        n += 1
                        if limit is not None and n >= limit:
                            return
