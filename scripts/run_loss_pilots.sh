#!/usr/bin/env bash
set -euo pipefail

# Two controlled loss pilots: member 0, tasks 0-1, sequentially.
# The existing baseline configuration/artifacts are never modified.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p outputs

MODE="${1:-dry-run}"
if [[ "$MODE" != "dry-run" && "$MODE" != "--execute" ]]; then
  echo "Usage: bash scripts/run_loss_pilots.sh [dry-run|--execute]" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONUNBUFFERED=1

CONFIGS=(
  "configs/pilot_er_p4_t01.json"
  "configs/pilot_hybrid_p4_t01.json"
)

for CONFIG in "${CONFIGS[@]}"; do
  echo "=== loss pilot: ${CONFIG} | member=0 | tasks=0-1 ==="
  ARGS=(
    "$PYTHON_BIN" src/training/fold_ensemble3_pilot.py
    --config "$CONFIG"
    --member-id 0
  )
  if [[ "$MODE" == "--execute" ]]; then
    "${ARGS[@]}" --execute
  else
    "${ARGS[@]}"
  fi
done

echo "Loss pilots finished. Outputs: outputs/loss_pilot_er_p4_t01 and outputs/loss_pilot_hybrid_p4_t01"
