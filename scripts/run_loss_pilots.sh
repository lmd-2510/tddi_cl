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

# The fold/preprocessing artifacts are often timestamped on the GPU server.
# Supply both roots to relocate inputs without changing the locked configs.
OVERRIDES=()
if [[ -n "${FOLD_ROOT:-}" || -n "${PREP_ROOT:-}" ]]; then
  : "${FOLD_ROOT:?Set FOLD_ROOT to the directory containing fold_assignments.parquet and fold_manifest.json}"
  : "${PREP_ROOT:?Set PREP_ROOT to the directory containing member_{0,1,2}/B/fold_preprocessing.json}"
  OVERRIDES+=(
    --fold-assignments "$FOLD_ROOT/fold_assignments.parquet"
    --fold-manifest "$FOLD_ROOT/fold_manifest.json"
    --preprocessing-root "$PREP_ROOT"
  )
fi

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
    "${ARGS[@]}" "${OVERRIDES[@]}" --execute
  else
    "${ARGS[@]}" "${OVERRIDES[@]}"
  fi
done

echo "Loss pilots finished. Outputs: outputs/loss_pilot_er_p4_t01 and outputs/loss_pilot_hybrid_p4_t01"
