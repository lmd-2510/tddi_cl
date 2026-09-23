#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_visualizations.sh RUN_ROOT [OUTDIR] [extra visualize args...]" >&2
  exit 2
fi

RUN_ROOT="$1"
OUTDIR="${2:-$RUN_ROOT/visualizations}"
if [[ $# -ge 2 ]]; then
  shift 2
else
  shift
fi

python scripts/visualize_cil_run.py \
  --run-root "$RUN_ROOT" \
  --outdir "$OUTDIR" \
  --ensemble \
  "$@"

echo "[OK] visualizations: $OUTDIR"
