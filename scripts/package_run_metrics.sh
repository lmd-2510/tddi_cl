#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash scripts/package_run_metrics.sh RUN_ROOT [ARCHIVE_PATH]" >&2
  exit 2
fi

RUN_ROOT="$1"
if [[ $# -eq 2 ]]; then
  python -m src.eval.package --run-root "$RUN_ROOT" --archive "$2"
else
  python -m src.eval.package --run-root "$RUN_ROOT"
fi
