#!/usr/bin/env bash
# P3 DER++ full 8-task, 3-member run. `start` detaches with nohup.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2
ACTION="${1:-start}"
PYTHON_BIN="${DERPP_PYTHON:-python}"
GPU_ID="${DERPP_GPU_ID:-0}"
CONFIG="$ROOT/configs/p3_derpp_full8_mem4.json"
MONITOR_ROOT="$ROOT/outputs/p3_experiments/derpp"
LATEST="$MONITOR_ROOT/latest.txt"

die() { echo "[STOP] $*" >&2; exit 2; }

find_fold_root() {
  if [[ -n "${DERPP_FOLD_ROOT:-}" ]]; then printf '%s\n' "$DERPP_FOLD_ROOT"; return; fi
  local preferred="$ROOT/study_assets/stratified_3fold_seed42"
  if [[ -s "$preferred/fold_assignments.parquet" && -s "$preferred/fold_manifest.json" ]]; then
    printf '%s\n' "$preferred"; return
  fi
  local -a found=()
  mapfile -t found < <(find "$ROOT/outputs" -type f -name fold_assignments.parquet -printf '%h\n' 2>/dev/null | sort -u)
  [[ ${#found[@]} -eq 1 && -s "${found[0]}/fold_manifest.json" ]] || return 1
  printf '%s\n' "${found[0]}"
}

find_prep_root() {
  if [[ -n "${DERPP_PREP_ROOT:-}" ]]; then printf '%s\n' "$DERPP_PREP_ROOT"; return; fi
  local preferred="$ROOT/study_assets/preprocessing_p3_seed0_fold42"
  if [[ -s "$preferred/member_0/B/fold_preprocessing.json" && -s "$preferred/member_1/B/fold_preprocessing.json" && -s "$preferred/member_2/B/fold_preprocessing.json" ]]; then
    printf '%s\n' "$preferred"; return
  fi
  return 1
}

if [[ "$ACTION" == "status" || "$ACTION" == "follow" ]]; then
  [[ -s "$LATEST" ]] || die "No DER++ run is registered."
  RUN_HOME="$(<"$LATEST")"
  [[ -s "$RUN_HOME/job.pid" ]] || die "Missing DER++ PID file."
  PID="$(<"$RUN_HOME/job.pid")"
  if [[ "$ACTION" == "follow" ]]; then
    tail -f "$RUN_HOME/training_and_offline.log"
  else
    if kill -0 "$PID" 2>/dev/null; then echo "[RUNNING] PID=$PID"; else echo "[STOPPED/FINISHED] PID=$PID"; fi
    if [[ -s "$RUN_HOME/output_root.txt" ]]; then
      OUTPUT_ROOT="$(<"$RUN_HOME/output_root.txt")"
      echo "[OUTPUT] $OUTPUT_ROOT"
      [[ -s "$OUTPUT_ROOT/experiment_status.txt" ]] && cat "$OUTPUT_ROOT/experiment_status.txt"
    fi
    [[ -s "$RUN_HOME/training_and_offline.log" ]] && tail -n 15 "$RUN_HOME/training_and_offline.log"
    [[ -s "$RUN_HOME/package.log" ]] && tail -n 5 "$RUN_HOME/package.log"
  fi
  exit 0
fi

FOLD_ROOT="$(find_fold_root)" || die "Cannot find one frozen fold root; set DERPP_FOLD_ROOT."
PREP_ROOT="$(find_prep_root)" || die "Cannot find P3 preprocessing; set DERPP_PREP_ROOT."
TAG="${2:-${DERPP_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}}"
RUN_HOME="$MONITOR_ROOT/run_$TAG"
OUT_ROOT="$ROOT/outputs/p3_derpp_full8_mem4_seed0_$TAG"
COMMON=(
  "$PYTHON_BIN" "$ROOT/scripts/run_p3_derpp_full.py"
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
    train_status=0
    "${COMMON[@]}" run > "$RUN_HOME/training_and_offline.log" 2>&1 || train_status=$?
    printf 'train_and_offline_exit=%s\n' "$train_status" > "$OUT_ROOT/experiment_status.txt"
    archive="$OUT_ROOT/review/p3_derpp_full8_mem4_review_$TAG.zip"
    package_status=0
    "$PYTHON_BIN" "$ROOT/scripts/package_p3_experiment_review.py" \
      --label "P3 DER++ online, fixed 178 head, 4 percent per member, full 8 tasks" \
      --run-root "$OUT_ROOT" --config "$CONFIG" \
      --status "train_and_offline_exit=$train_status" \
      --log "$RUN_HOME/training_and_offline.log" --archive "$archive" \
      > "$RUN_HOME/package.log" 2>&1 || package_status=$?
    if [[ "$package_status" -eq 0 ]]; then
      echo "[ZIP] $archive"
      echo "[SHA256] $archive.sha256"
    else
      echo "[WARN] Packaging failed; see $RUN_HOME/package.log" >&2
    fi
    printf 'package_exit=%s\n' "$package_status" >> "$OUT_ROOT/experiment_status.txt"
    echo "[STATUS] train_and_offline=$train_status package=$package_status"
    (( train_status == 0 && package_status == 0 ))
    ;;
  start)
    [[ ! -e "$RUN_HOME" && ! -e "$OUT_ROOT" ]] || die "Run tag already exists; choose a fresh tag."
    mkdir -p "$RUN_HOME"
    "${COMMON[@]}" check > "$RUN_HOME/preflight.log" 2>&1 || {
      tail -n 30 "$RUN_HOME/preflight.log" >&2
      die "DER++ preflight failed."
    }
    printf '%s\n' "$OUT_ROOT" > "$RUN_HOME/output_root.txt"
    printf '%s\n' "$RUN_HOME" > "$LATEST"
    nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 \
      DERPP_FOLD_ROOT="$FOLD_ROOT" DERPP_PREP_ROOT="$PREP_ROOT" DERPP_PYTHON="$PYTHON_BIN" \
      bash "$ROOT/scripts/run_p3_derpp_full.sh" run "$TAG" \
      > "$RUN_HOME/nohup.log" 2>&1 < /dev/null &
    echo $! > "$RUN_HOME/job.pid"
    echo "[STARTED] PID=$(<"$RUN_HOME/job.pid"); P3 DER++, members 0→1→2, tasks 0–7."
    echo "[LOG] $RUN_HOME/training_and_offline.log"
    echo "[OUTPUT] $OUT_ROOT"
    ;;
  *)
    die "Usage: bash scripts/run_p3_derpp_full.sh [check|start|run|status|follow]"
    ;;
esac
