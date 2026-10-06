#!/usr/bin/env bash
# End-to-end pilot: build mixture -> overlap audit -> train slot model + letter baseline -> calibrate -> evaluate.
# Validation / heldout only. The test split is NOT touched here (use evaluate --split test --final, once, deliberately).
#
#   scripts/run_pilot.sh                 # full run on a single GPU node; run from any folder, with the project environment active
#   LIMIT=300 scripts/run_pilot.sh       # smoke build (limits every source/split)
#   SKIP_BUILD=1 scripts/run_pilot.sh    # reuse an existing mixture
set -euo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
PY="${PY:-python}"   # Python of the active environment (uv .venv or conda); override with PY=/path/to/python
DATA="${DATA:-$HOME/mimir-decide-data/mixture-v0}"
RUNS="${RUNS:-$HOME/mimir-decide-data/runs}"

if [[ -z "${SKIP_BUILD:-}" ]]; then
  $PY -m mimir_decide.build_mixture --config configs/data.yaml --output_dir "$DATA" ${LIMIT:+--limit $LIMIT}
fi
$PY -m mimir_decide.audit_mimir_overlap --data_dir "$DATA"

for NAME in slot baseline; do
  RUN="$RUNS/${NAME}-v0"
  $PY -m mimir_decide.train --config "configs/train_${NAME}.yaml" --set data_dir="$DATA" run_dir="$RUN" ${EXTRA_SET:-}
  $PY -m mimir_decide.calibrate --run_dir "$RUN" --data_dir "$DATA"
  $PY -m mimir_decide.evaluate  --run_dir "$RUN" --data_dir "$DATA" --split validation
  [[ -f "$DATA/heldout_tasks/heldout.parquet" ]] && $PY -m mimir_decide.evaluate --run_dir "$RUN" --data_dir "$DATA" --split heldout
done
echo "pilot finished. Compare $RUNS/slot-v0/eval/*.json with $RUNS/baseline-v0/eval/*.json"
