#!/usr/bin/env bash
# Run our models through the Decision Index kit (E15, D29) and copy the small score files into the repository.
#
#   cd <decision-index checkout>      # the kit must be installed in the same environment as mimir_decide
#   bash /work/dfm/lukasgp/mimir-decide/scripts/run_decision_index.sh ROWS OUT_DIR MAX_LEN run:checkpoint [run:checkpoint ...]
#
#   ROWS        a rows file (sample-2000.jsonl.gz) or '-' for the full suite of the kit
#   OUT_DIR     results go to OUT_DIR/di-<run>/ (results.jsonl.gz is large: keep it outside the repository)
#   MAX_LEN     maximum prompt length in tokens; requests that do not fit are refused, nothing is truncated (D29)
#   run:checkpoint   for example baseline-v0:best  letter-zeroshot:zero-shot
# Environment: REPO (default /work/dfm/lukasgp/mimir-decide), BATCH (questions per forward pass, default 8), TAG (suffix of
# the copied score files, for example sample2000), EDITION (suite edition of the kit, default 0.3; 0.2 for the rebuilt suite).
set -u
ROWS=$1; OUT=$2; MAXLEN=$3; shift 3
REPO=${REPO:-/work/dfm/lukasgp/mimir-decide}; BATCH=${BATCH:-8}; TAG=${TAG:-run}; EDITION=${EDITION:-0.3}
mkdir -p "$OUT" "$REPO/results/decision-index"
for m in "$@"; do
  run=${m%%:*}; ck=${m##*:}
  dir="$OUT/di-$run"
  echo "=== $run ($ck) -> $dir   max_len=$MAXLEN   $(date)"
  rows_args=(); [ "$ROWS" != "-" ] && rows_args=(--rows "$ROWS")
  python -u -m decision_index run --engine mimir_decide.decision_index_engine:MimirDecideEngine \
    --option "run_dir=$REPO/runs/$run" --option "checkpoint=$ck" --option "max_len=$MAXLEN" --option "batch_size=$BATCH" \
    --edition "$EDITION" "${rows_args[@]}" --out "$dir" 2>&1 | tail -n 3
  python -m decision_index score --edition "$EDITION" --results "$dir/results.jsonl.gz" --out "$dir" 2>&1 | tail -n 3
  [ -f "$dir/scores.json" ] && cp -f "$dir/scores.json" "$REPO/results/decision-index/$run.$TAG.scores.json"
done
