#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CIL_DDINET_FULL_RUN_METHOD="upcl"
exec bash "$PROJECT_ROOT/scripts/run_ewc_ddinet_head_to_tail_full.sh" "$@"
