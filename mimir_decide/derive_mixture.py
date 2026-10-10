"""Derive a mixture with extra held-out sources from an existing mixture (experiment E12). No rebuild, no new downloads.

    python -m mimir_decide.derive_mixture --src_dir $DATA --dst_dir $DATA_UNSEEN --sources_file configs/heldout_unseen_labels.yaml

For every source in `held_out_sources`:
  * its rows are removed from train.parquet;
  * its validation, calibration and test rows are moved to heldout_tasks/heldout.parquet, which keeps the old
    held-out rows (for example massive/nb-NO), exactly as build_mixture does for held-out tasks.
Everything else is copied. Only train rows are removed, so no leakage guard is affected. The new
mixture_manifest.json records where the mixture comes from, so each run that trains on it is traceable.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

EVAL_FILES = ["validation.parquet", "calib.parquet", "eval/test/test.parquet"]


def _split(table: pa.Table, sources: list[str]):
    mask = pc.is_in(table["source"], value_set=pa.array(sources, type=pa.string()))
    return table.filter(pc.invert(mask)), table.filter(mask)


def derive(src: Path, dst: Path, sources: list[str]) -> dict:
    src, dst = Path(src).expanduser(), Path(dst).expanduser()
    assert src.resolve() != dst.resolve(), "dst_dir must differ from src_dir"
    assert not dst.exists() or not any(dst.iterdir()), f"{dst} exists and is not empty"
    assert sources, "no sources to hold out"
    (dst / "heldout_tasks").mkdir(parents=True, exist_ok=True)
    (dst / "eval" / "test").mkdir(parents=True, exist_ok=True)

    train = pq.read_table(src / "train.parquet")
    present = set(train["source"].to_pylist()) | {s for f in EVAL_FILES if (src / f).exists()
                                                  for s in pq.read_table(src / f, columns=["source"])["source"].to_pylist()}
    missing = [s for s in sources if s not in present]
    assert not missing, f"sources not found in the mixture: {missing}"
    keep_train, drop_train = _split(train, sources)
    pq.write_table(keep_train, dst / "train.parquet", compression="zstd")

    moved = [] ; counts = {"train_removed": drop_train.num_rows, "train_kept": keep_train.num_rows}
    for f in EVAL_FILES:
        if not (src / f).exists():
            continue
        keep, mv = _split(pq.read_table(src / f), sources)
        pq.write_table(keep, dst / f, compression="zstd")
        moved.append(mv)
        counts[f"{Path(f).stem}_kept"], counts[f"{Path(f).stem}_moved"] = keep.num_rows, mv.num_rows
    old = src / "heldout_tasks" / "heldout.parquet"
    parts = ([pq.read_table(old)] if old.exists() else []) + moved
    held = pa.concat_tables(parts, promote_options="default") if parts else pa.table({})
    pq.write_table(held, dst / "heldout_tasks" / "heldout.parquet", compression="zstd")
    counts["heldout_total"] = held.num_rows

    if (src / "reports").exists():
        shutil.copytree(src / "reports", dst / "reports")
    manifest = json.loads((src / "mixture_manifest.json").read_text()) if (src / "mixture_manifest.json").exists() else {}
    manifest["derived"] = {"from": str(src), "extra_held_out_sources": sources, "counts": counts,
                           "note": "train rows of these sources removed; their eval rows moved to heldout_tasks/heldout.parquet"}
    manifest["counts"] = {**manifest.get("counts", {}), "train": keep_train.num_rows, "heldout": held.num_rows}
    (dst / "mixture_manifest.json").write_text(json.dumps(manifest, indent=1))

    # safety checks on what was written
    for f in ["train.parquet", *[x for x in EVAL_FILES if (dst / x).exists()]]:
        left = set(pq.read_table(dst / f, columns=["source"])["source"].to_pylist()) & set(sources)
        assert not left, f"{f} still contains held-out sources {left}"
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--dst_dir", required=True)
    ap.add_argument("--sources_file", required=True, help="YAML with held_out_sources (exact source names)")
    a = ap.parse_args(argv)
    sources = yaml.safe_load(open(a.sources_file))["held_out_sources"]
    print(json.dumps(derive(Path(a.src_dir), Path(a.dst_dir), sources), indent=1))


if __name__ == "__main__":
    main()
