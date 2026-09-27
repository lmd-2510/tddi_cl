#!/usr/bin/env bash
# P3 DER++ equal-class-buffer pilot: member 0, all eight tasks, nohup + review ZIP.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2
ACTION="${1:-start}"
PYTHON_BIN="${DERPP_PILOT_PYTHON:-python}"
GPU_ID="${DERPP_PILOT_GPU_ID:-0}"
CONFIG="$ROOT/configs/p3_derpp_equal_buffer_pilot.json"
MONITOR_ROOT="$ROOT/outputs/p3_experiments/derpp_equal_buffer_pilot"
LATEST="$MONITOR_ROOT/latest.txt"

die() { echo "[STOP] $*" >&2; exit 2; }

find_fold_root() {
  if [[ -n "${DERPP_PILOT_FOLD_ROOT:-}" ]]; then printf '%s\n' "$DERPP_PILOT_FOLD_ROOT"; return; fi
  local preferred="$ROOT/outputs/fold_preparation_seed42_20260916_230549/folds"
  if [[ -s "$preferred/fold_assignments.parquet" && -s "$preferred/fold_manifest.json" ]]; then
    printf '%s\n' "$preferred"; return
  fi
  local -a found=()
  mapfile -t found < <(find "$ROOT/outputs" -type f -name fold_assignments.parquet -printf '%h\n' 2>/dev/null | sort -u)
  [[ ${#found[@]} -eq 1 && -s "${found[0]}/fold_manifest.json" ]] || return 1
  printf '%s\n' "${found[0]}"
}

find_prep_root() {
  if [[ -n "${DERPP_PILOT_PREP_ROOT:-}" ]]; then printf '%s\n' "$DERPP_PILOT_PREP_ROOT"; return; fi
  local preferred="$ROOT/study_assets/preprocessing_p3_seed0_fold42"
  [[ -s "$preferred/member_0/B/fold_preprocessing.json" ]] || return 1
  printf '%s\n' "$preferred"
}

if [[ "$ACTION" == "status" || "$ACTION" == "follow" ]]; then
  [[ -s "$LATEST" ]] || die "No pilot run is registered."
  RUN_HOME="$(<"$LATEST")"
  [[ -s "$RUN_HOME/job.pid" ]] || die "Missing pilot PID file."
  PID="$(<"$RUN_HOME/job.pid")"
  if [[ "$ACTION" == "follow" ]]; then
    tail -f "$RUN_HOME/training_and_visualizations.log"
  else
    if kill -0 "$PID" 2>/dev/null; then echo "[RUNNING] PID=$PID"; else echo "[STOPPED/FINISHED] PID=$PID"; fi
    if [[ -s "$RUN_HOME/output_root.txt" ]]; then
      OUTPUT_ROOT="$(<"$RUN_HOME/output_root.txt")"
      echo "[OUTPUT] $OUTPUT_ROOT"
      [[ -s "$OUTPUT_ROOT/experiment_status.txt" ]] && cat "$OUTPUT_ROOT/experiment_status.txt"
    fi
    [[ -s "$RUN_HOME/training_and_visualizations.log" ]] && tail -n 15 "$RUN_HOME/training_and_visualizations.log"
    [[ -s "$RUN_HOME/package.log" ]] && tail -n 5 "$RUN_HOME/package.log"
  fi
  exit 0
fi

FOLD_ROOT="$(find_fold_root)" || die "Cannot find frozen fold root; set DERPP_PILOT_FOLD_ROOT."
PREP_ROOT="$(find_prep_root)" || die "Cannot find P3 member-0 preprocessing; set DERPP_PILOT_PREP_ROOT."
TAG="${2:-${DERPP_PILOT_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}}"
RUN_HOME="$MONITOR_ROOT/run_$TAG"
OUT_ROOT="$ROOT/outputs/p3_derpp_equal_buffer_member0_$TAG"
COMMON=(
  "$PYTHON_BIN" "$ROOT/scripts/run_p3_derpp_equal_buffer_pilot.py"
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
    mkdir -p "$RUN_HOME" "$OUT_ROOT"
    printf '%s\n' "$OUT_ROOT" > "$RUN_HOME/output_root.txt"
    run_status=0
    "${COMMON[@]}" run > "$RUN_HOME/training_and_visualizations.log" 2>&1 || run_status=$?
    printf 'train_and_visualizations_exit=%s\n' "$run_status" > "$OUT_ROOT/experiment_status.txt"
    archive="$OUT_ROOT/review/p3_derpp_equal_buffer_member0_$TAG.zip"
    package_status=0
    "$PYTHON_BIN" "$ROOT/scripts/package_p3_experiment_review.py" \
      --label "P3 DER++ equal-class-buffer member-0 pilot; full eight tasks; no ensemble" \
      --run-root "$OUT_ROOT" --config "$CONFIG" \
      --status "train_and_visualizations_exit=$run_status" \
      --log "$RUN_HOME/training_and_visualizations.log" --archive "$archive" \
      > "$RUN_HOME/package.log" 2>&1 || package_status=$?
    printf 'package_exit=%s\n' "$package_status" >> "$OUT_ROOT/experiment_status.txt"
    if [[ "$package_status" -eq 0 ]]; then
      echo "[ZIP] $archive"
      echo "[SHA256] $archive.sha256"
    else
      echo "[WARN] Packaging failed; see $RUN_HOME/package.log" >&2
    fi
    echo "[STATUS] train_and_visualizations=$run_status package=$package_status"
    (( run_status == 0 && package_status == 0 ))
    ;;
  start)
    [[ ! -e "$RUN_HOME" && ! -e "$OUT_ROOT" ]] || die "Run tag already exists; choose a fresh tag."
    mkdir -p "$RUN_HOME"
    "${COMMON[@]}" check > "$RUN_HOME/preflight.log" 2>&1 || {
      tail -n 30 "$RUN_HOME/preflight.log" >&2
      die "Pilot preflight failed; no training was started."
    }
    printf '%s\n' "$OUT_ROOT" > "$RUN_HOME/output_root.txt"
    printf '%s\n' "$RUN_HOME" > "$LATEST"
    nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 \
      DERPP_PILOT_FOLD_ROOT="$FOLD_ROOT" DERPP_PILOT_PREP_ROOT="$PREP_ROOT" \
      DERPP_PILOT_PYTHON="$PYTHON_BIN" \
      bash "$ROOT/scripts/run_p3_derpp_equal_buffer_pilot.sh" run "$TAG" \
      > "$RUN_HOME/nohup.log" 2>&1 < /dev/null &
    echo $! > "$RUN_HOME/job.pid"
    echo "[STARTED] PID=$(<"$RUN_HOME/job.pid"); member 0 only, tasks 0–7."
    echo "[LOG] $RUN_HOME/training_and_visualizations.log"
    echo "[OUTPUT] $OUT_ROOT"
    ;;
  *)
    die "Usage: bash scripts/run_p3_derpp_equal_buffer_pilot.sh [check|start|run|status|follow]"
    ;;
esac
