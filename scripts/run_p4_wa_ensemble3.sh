#!/usr/bin/env bash
set -euo pipefail

# One-command P4 + Weight Aligning study. With no argument this prepares any
# missing member scaler, then launches members 0 -> 1 -> 2, ensemble/UE,
# thresholds and the final report through the shared resumable controller.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export P4_FULL_CONFIG="$REPO_ROOT/configs/full_tddi_p4_fold_ensemble3_seed0_wa.json"
export P4_FULL_ROOT="$REPO_ROOT/outputs/stratified_ensemble3/full_p4_seed0_8tasks_wa"
export P4_MONITOR_ROOT="$REPO_ROOT/outputs/stratified_ensemble3_monitor/full_p4_seed0_8tasks_wa"
export P4_CONTROLLER_SCRIPT="scripts/run_p4_wa_ensemble3.sh"

if (( $# == 0 )); then
  bash "$SCRIPT_DIR/run_p4_ensemble3.sh" prepare
  exec bash "$SCRIPT_DIR/run_p4_ensemble3.sh" start
fi

exec bash "$SCRIPT_DIR/run_p4_ensemble3.sh" "$@"
