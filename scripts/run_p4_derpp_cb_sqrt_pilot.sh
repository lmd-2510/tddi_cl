#!/usr/bin/env bash
# P4 DER++ softened square-root replay pilot: member 0, tasks 0-7, review ZIP.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2
ACTION="${1:-start}"
PYTHON_BIN="${P4_SQRT_PYTHON:-python}"
GPU_ID="${P4_SQRT_GPU_ID:-0}"
CONFIG="$ROOT/configs/p4_derpp_cb_sqrt_replay_pilot.json"
MONITOR="$ROOT/outputs/p4_derpp_cb_sqrt_pilot_monitor"
LATEST="$MONITOR/latest.txt"

die() { echo "[STOP] $*" >&2; exit 2; }

find_fold_root() {
  if [[ -n "${P4_SQRT_FOLD_ROOT:-}" ]]; then printf '%s\n' "$P4_SQRT_FOLD_ROOT"; return; fi
  local preferred="$ROOT/outputs/fold_preparation_seed42_20260916_230549/folds"
  if [[ -s "$preferred/fold_assignments.parquet" && -s "$preferred/fold_manifest.json" ]]; then
    printf '%s\n' "$preferred"; return
  fi
  local -a found=()
  mapfile -t found < <(find "$ROOT/outputs" -type f -name fold_assignments.parquet -printf '%h\n' 2>/dev/null | sort -u)
  [[ ${#found[@]} -eq 1 && -s "${found[0]}/fold_manifest.json" ]] || return 1
  printf '%s\n' "${found[0]}"
}

latest_home() {
  [[ -s "$LATEST" ]] || die "No P4 sqrt-replay pilot is registered."
  local value
  value="$(<"$LATEST")"
  [[ -d "$value" && "$value" == "$MONITOR"/run_* ]] || die "Invalid latest pointer: $value"
  printf '%s\n' "$value"
}

FOLD_ROOT="$(find_fold_root)" || die "Cannot resolve fold root; set P4_SQRT_FOLD_ROOT."

launch() {
  local run_home="$1" tag="$2" out="$3"
  nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 \
    P4_SQRT_FOLD_ROOT="$FOLD_ROOT" P4_SQRT_PYTHON="$PYTHON_BIN" \
    bash "$ROOT/scripts/run_p4_derpp_cb_sqrt_pilot.sh" run "$tag" \
    > "$run_home/nohup.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$run_home/job.pid"
  echo "[STARTED] PID=$(<"$run_home/job.pid"); member 0 only, tasks 0-7."
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
      if kill -0 "$PID" 2>/dev/null; then echo "[RUNNING] PID=$PID"
      elif [[ -s "$RUN_HOME/exit_code.txt" ]]; then echo "[FINISHED] exit_code=$(<"$RUN_HOME/exit_code.txt")"
      else echo "[STOPPED/UNKNOWN] PID=$PID"; fi
      echo "[OUTPUT] $OUT_ROOT"
      if [[ -s "$OUT_ROOT/pilot_manifest.json" ]]; then
        "$PYTHON_BIN" - "$OUT_ROOT/pilot_manifest.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"Member0 Task6 Macro-F1: {d['test_macro_f1_task6']:.6f}")
print(f"Member0 Task7 Macro-F1: {d['test_macro_f1_task7']:.6f}")
print(f"Task6->7 delta: {d['task6_to_task7_delta']:+.6f}")
PY
      fi
      [[ -s "$RUN_HOME/main.log" ]] && tail -n 15 "$RUN_HOME/main.log"
    fi
    exit 0
    ;;
esac

TAG="${2:-${P4_SQRT_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}}"
RUN_HOME="$MONITOR/run_$TAG"
OUT_ROOT="$ROOT/outputs/p4_derpp_cb_sqrt_pilot/$TAG"
PREP_ROOT="$OUT_ROOT/config/preprocessing"
COMMON=(
  "$PYTHON_BIN" "$ROOT/scripts/run_p4_derpp_cb_sqrt_pilot.py"
  --config "$CONFIG" --train "$ROOT/train_extracted.parquet"
  --validation "$ROOT/validation_extracted.parquet" --test "$ROOT/test_extracted.parquet"
  --feature-cols "$ROOT/study_assets/data_schema/feature_columns.json"
  --fold-root "$FOLD_ROOT" --preprocessing-root "$PREP_ROOT"
  --output-root "$OUT_ROOT" --python "$PYTHON_BIN" --device cuda
)

case "$ACTION" in
  check)
    "${COMMON[@]}" check
    ;;
  run)
    mkdir -p "$RUN_HOME" "$OUT_ROOT/logs"
    printf '%s\n' "$OUT_ROOT" > "$RUN_HOME/output_root.txt"
    status=0
    "${COMMON[@]}" run >> "$RUN_HOME/main.log" 2>&1 || status=$?
    archive="$OUT_ROOT/review/p4_derpp_cb_sqrt_member0_${TAG}_review.zip"
    package_status=0
    "$PYTHON_BIN" "$ROOT/scripts/package_p3_experiment_review.py" \
      --label "P4 DER++ equal-buffer square-root-replay member-0 pilot" \
      --run-root "$OUT_ROOT" --config "$CONFIG" \
      --status "pilot_exit=$status; member0_only=true; offline_ensemble=false" \
      --log "$RUN_HOME/main.log" --archive "$archive" \
      >> "$RUN_HOME/package.log" 2>&1 || package_status=$?
    printf '%s\n' "$status" > "$OUT_ROOT/runner_exit_code.txt"
    printf '%s\n' "$package_status" > "$OUT_ROOT/package_exit_code.txt"
    final_status="$status"
    (( final_status == 0 && package_status != 0 )) && final_status="$package_status"
    printf '%s\n' "$final_status" > "$RUN_HOME/exit_code.txt"
    if [[ "$package_status" -eq 0 ]]; then
      echo "[ZIP] $archive"
      echo "[SHA256] $archive.sha256"
    else
      echo "[WARN] Packaging failed; inspect $RUN_HOME/package.log" >&2
    fi
    exit "$final_status"
    ;;
  start)
    [[ ! -e "$RUN_HOME" && ! -e "$OUT_ROOT" ]] || die "Run ID already exists: $TAG"
    mkdir -p "$RUN_HOME"
    "${COMMON[@]}" check > "$RUN_HOME/preflight.log" 2>&1 || {
      tail -n 40 "$RUN_HOME/preflight.log" >&2
      die "Preflight failed; no preprocessing or training started."
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
      die "Pilot is already running: PID=$(<"$RUN_HOME/job.pid")"
    fi
    if [[ -s "$OUT_ROOT/pilot_manifest.json" ]]; then
      echo "[OK] Pilot is already complete: $OUT_ROOT"
      exit 0
    fi
    launch "$RUN_HOME" "$TAG" "$OUT_ROOT"
    ;;
  *)
    die "Usage: bash scripts/run_p4_derpp_cb_sqrt_pilot.sh [check|start|resume|status|follow]"
    ;;
esac
