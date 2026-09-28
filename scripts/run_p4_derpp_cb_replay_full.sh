#!/usr/bin/env bash
# One-command P4 DER++ class-balanced storage + class-uniform replay full run.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2
ACTION="${1:-start}"
PYTHON_BIN="${P4_DERPP_PYTHON:-python}"
GPU_ID="${P4_DERPP_GPU_ID:-0}"
CONFIG="$ROOT/configs/p4_derpp_cb_class_uniform_full8.json"
TASK="$ROOT/study_assets/task_protocols/constrained_mass_balanced_seed0_tasks.json"
MONITOR="$ROOT/outputs/p4_derpp_cb_class_uniform_monitor"
LATEST="$MONITOR/latest.txt"

die() { echo "[STOP] $*" >&2; exit 2; }

find_fold_root() {
  if [[ -n "${P4_DERPP_FOLD_ROOT:-}" ]]; then printf '%s\n' "$P4_DERPP_FOLD_ROOT"; return; fi
  local -a found=()
  mapfile -t found < <(find "$ROOT/outputs" -type f -name fold_assignments.parquet -printf '%h\n' 2>/dev/null | sort -u)
  [[ ${#found[@]} -eq 1 && -s "${found[0]}/fold_manifest.json" ]] || return 1
  printf '%s\n' "${found[0]}"
}

latest_home() {
  [[ -s "$LATEST" ]] || die "No P4 DER++ CB run is registered."
  local value
  value="$(<"$LATEST")"
  [[ -d "$value" && "$value" == "$MONITOR"/run_* ]] || die "Invalid latest run pointer: $value"
  printf '%s\n' "$value"
}

FOLD_ROOT="$(find_fold_root)" || die "Cannot resolve one fold root; export P4_DERPP_FOLD_ROOT=/absolute/path/to/folds."

launch() {
  local run_home="$1" tag="$2" out="$3"
  nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 \
    P4_DERPP_FOLD_ROOT="$FOLD_ROOT" P4_DERPP_PYTHON="$PYTHON_BIN" \
    bash "$ROOT/scripts/run_p4_derpp_cb_replay_full.sh" run "$tag" \
    > "$run_home/nohup.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$run_home/job.pid"
  echo "[STARTED] PID=$(<"$run_home/job.pid")"
  echo "[OUTPUT] $out"
  echo "[LOG] $run_home/main.log"
}

case "$ACTION" in
  status|follow)
    RUN_HOME="$(latest_home)"
    OUT_ROOT="$(<"$RUN_HOME/output_root.txt")"
    if [[ "$ACTION" == "follow" ]]; then
      touch "$RUN_HOME/main.log"
      tail -f "$RUN_HOME/main.log"
    else
      PID="$(<"$RUN_HOME/job.pid")"
      if kill -0 "$PID" 2>/dev/null; then echo "[RUNNING] PID=$PID";
      elif [[ -s "$RUN_HOME/exit_code.txt" ]]; then echo "[FINISHED] exit_code=$(<"$RUN_HOME/exit_code.txt")";
      else echo "[STOPPED/UNKNOWN] PID=$PID"; fi
      "$PYTHON_BIN" "$ROOT/scripts/run_p4_derpp_cb_replay_full.py" status \
        --config "$CONFIG" --task-file "$TASK" --train "$ROOT/train_extracted.parquet" \
        --validation "$ROOT/validation_extracted.parquet" --test "$ROOT/test_extracted.parquet" \
        --feature-cols "$ROOT/study_assets/data_schema/feature_columns.json" \
        --fold-root "$FOLD_ROOT" --preprocessing-root "$OUT_ROOT/config/preprocessing" \
        --output-root "$OUT_ROOT" --python "$PYTHON_BIN" --device cuda
      [[ -s "$RUN_HOME/main.log" ]] && tail -n 12 "$RUN_HOME/main.log"
    fi
    exit 0
    ;;
esac

TAG="${2:-${P4_DERPP_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}}"
RUN_HOME="$MONITOR/run_$TAG"
OUT_ROOT="$ROOT/outputs/p4_derpp_cb_class_uniform/$TAG"
COMMON=(
  "$PYTHON_BIN" "$ROOT/scripts/run_p4_derpp_cb_replay_full.py"
  --config "$CONFIG" --task-file "$TASK"
  --train "$ROOT/train_extracted.parquet" --validation "$ROOT/validation_extracted.parquet"
  --test "$ROOT/test_extracted.parquet"
  --feature-cols "$ROOT/study_assets/data_schema/feature_columns.json"
  --fold-root "$FOLD_ROOT" --preprocessing-root "$OUT_ROOT/config/preprocessing"
  --output-root "$OUT_ROOT" --python "$PYTHON_BIN" --device cuda
)

case "$ACTION" in
  check)
    "${COMMON[@]}" check
    ;;
  run)
    mkdir -p "$RUN_HOME" "$OUT_ROOT/logs"
    printf '%s\n' "$OUT_ROOT" > "$RUN_HOME/output_root.txt"
    attempt=1
    if [[ -s "$OUT_ROOT/runner_attempts.txt" ]]; then
      attempt=$(( $(wc -l < "$OUT_ROOT/runner_attempts.txt") + 1 ))
    fi
    printf 'attempt=%s started_utc=%s\n' "$attempt" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      >> "$OUT_ROOT/runner_attempts.txt"
    printf '\n[RUNNER_ATTEMPT] %s\n' "$attempt" >> "$RUN_HOME/main.log"
    status=0
    "${COMMON[@]}" run >> "$RUN_HOME/main.log" 2>&1 || status=$?
    printf '%s\n' "$status" > "$RUN_HOME/exit_code.txt"
    printf '%s\n' "$status" > "$OUT_ROOT/runner_exit_code.txt"
    exit "$status"
    ;;
  start)
    [[ ! -e "$RUN_HOME" && ! -e "$OUT_ROOT" ]] || die "Run ID already exists: $TAG"
    mkdir -p "$RUN_HOME"
    "${COMMON[@]}" check > "$RUN_HOME/preflight.log" 2>&1 || {
      tail -n 40 "$RUN_HOME/preflight.log" >&2
      die "Preflight failed."
    }
    printf '%s\n' "$OUT_ROOT" > "$RUN_HOME/output_root.txt"
    printf '%s\n' "$RUN_HOME" > "$LATEST"
    launch "$RUN_HOME" "$TAG" "$OUT_ROOT"
    ;;
  resume)
    RUN_HOME="$(latest_home)"
    OUT_ROOT="$(<"$RUN_HOME/output_root.txt")"
    TAG="${RUN_HOME##*/run_}"
    if [[ -s "$RUN_HOME/job.pid" ]] && kill -0 "$(<"$RUN_HOME/job.pid")" 2>/dev/null; then
      die "Run is already live: PID=$(<"$RUN_HOME/job.pid")"
    fi
    if [[ -s "$OUT_ROOT/final_manifest.json" ]] && grep -q '"final_status": "completed"' "$OUT_ROOT/final_manifest.json"; then
      echo "[OK] Run is already completed: $OUT_ROOT"
      exit 0
    fi
    launch "$RUN_HOME" "$TAG" "$OUT_ROOT"
    ;;
  *)
    die "Usage: bash scripts/run_p4_derpp_cb_replay_full.sh [check|start|resume|status|follow]"
    ;;
esac
