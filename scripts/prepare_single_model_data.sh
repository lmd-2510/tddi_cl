#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-}"
CONFIG="${2:-configs/ordered_preprocessing.json}"
THROUGH_TASK="${THROUGH_TASK:-7}"
CONFIG_PATH="$ROOT/$CONFIG"
NAME="$(basename "$CONFIG" .json)"
RUN_HOME="$ROOT/outputs/preprocessing_monitor/$NAME"
mkdir -p "$RUN_HOME"

case "$ACTION" in
  check)
    python "$ROOT/scripts/prepare_single_model_data.py" --config "$CONFIG_PATH" \
      --through-task "$THROUGH_TASK" --check
    ;;
  run)
    exec python "$ROOT/scripts/prepare_single_model_data.py" --config "$CONFIG_PATH" \
      --through-task "$THROUGH_TASK"
    ;;
  start)
    nohup python "$ROOT/scripts/prepare_single_model_data.py" --config "$CONFIG_PATH" \
      --through-task "$THROUGH_TASK" >"$RUN_HOME/nohup.log" 2>&1 < /dev/null &
    printf '%s\n' "$!" >"$RUN_HOME/job.pid"
    echo "[STARTED] PID=$!"
    echo "[LOG] $RUN_HOME/nohup.log"
    ;;
  status)
    PID="$(cat "$RUN_HOME/job.pid" 2>/dev/null || true)"
    if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
      echo "[RUNNING] PID=$PID"
    else
      echo "[NOT RUNNING]"
    fi
    tail -n 30 "$RUN_HOME/nohup.log" 2>/dev/null || true
    ;;
  follow)
    tail -f "$RUN_HOME/nohup.log"
    ;;
  *)
    echo "Usage: bash scripts/prepare_single_model_data.sh {check|run|start|status|follow} [config]" >&2
    exit 2
    ;;
esac
