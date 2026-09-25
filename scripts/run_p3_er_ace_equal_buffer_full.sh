#!/usr/bin/env bash
# Full P3 ER-ACE with equal-class memory. Delegates the established detached
# launcher, offline ensemble, task-6/7 diagnostics, and review ZIP workflow.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export P3_CONFIG="$ROOT/configs/p3_er_ace_equal_buffer_full8_e30_mem4.json"
export P3_EXPERIMENT_SLUG="p3_er_ace_equal_buffer_full8_e30_mem4"
export P3_DISPLAY_LABEL="P3 ER-ACE + equal-class buffer full8 e30 mem4"
export P3_LOSS_LABEL="ER-ACE (current CE masked to current task classes + replay CE over all seen classes)"
export P3_MONITOR_ROOT="$ROOT/outputs/p3_experiments/er_ace_equal_buffer"

exec bash "$ROOT/scripts/run_p3_hybrid_equal_buffer_full.sh" "$@"
