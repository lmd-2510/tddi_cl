#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CIL_FULL_RUN_METHOD="agem"
export CIL_FULL_RUN_LAUNCHER="scripts/run_agem_head_to_tail_full.sh"
exec bash "$PROJECT_ROOT/scripts/run_ewc_head_to_tail_full.sh" "$@"
