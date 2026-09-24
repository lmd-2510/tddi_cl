#!/usr/bin/env bash
set -euo pipefail

# One-member pilot for Class-Balanced Hybrid:
# class-balanced Focal on current-task samples, ordinary CE on replay,
# no logit/feature distillation. All other settings match the e30 P3 baseline.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${CB_HYB_PYTHON:-python}"
GPU="${CB_HYB_GPU_ID:-0}"
CONFIG="${CB_HYB_CONFIG:-$ROOT/configs/p3_cb_hybrid_full8_e30_mem4_pilot.json}"
OUT="${CB_HYB_OUT:-$ROOT/outputs/p3_cb_hybrid_full8_e30_mem4_pilot_seed0}"
MONITOR="${CB_HYB_MONITOR:-$ROOT/outputs/p3_cb_hybrid_pilot_monitor}"
TASK="$ROOT/study_assets/task_protocols/tail_to_head_tasks.json"
THRESHOLD="$ROOT/configs/eval_tddi_p3_ensemble_entropy_threshold.json"

die() { echo "[STOP] $*" >&2; exit 1; }

find_fold_root() {
  if [[ -n "${CB_HYB_FOLD_ROOT:-}" ]]; then
    printf '%s\n' "$CB_HYB_FOLD_ROOT"
    return
  fi
  local preferred="$ROOT/study_assets/stratified_3fold_seed42"
  if [[ -s "$preferred/fold_assignments.parquet" && -s "$preferred/fold_manifest.json" ]]; then
    printf '%s\n' "$preferred"
    return
  fi
  mapfile -t found < <(find "$ROOT/outputs" -type f -name fold_assignments.parquet -printf '%h\n' 2>/dev/null | sort -u)
  (( ${#found[@]} == 1 )) || die "Set CB_HYB_FOLD_ROOT to the directory containing fold_assignments.parquet and fold_manifest.json"
  printf '%s\n' "${found[0]}"
}

FOLD_ROOT="$(find_fold_root)"
PREP_ROOT="${CB_HYB_PREP_ROOT:-$ROOT/study_assets/preprocessing_p3_seed0_fold42}"
COMMON=(
  --config "$CONFIG"
  --train "$ROOT/train_extracted.parquet"
  --validation "$ROOT/validation_extracted.parquet"
  --test "$ROOT/test_extracted.parquet"
  --feature-cols "$ROOT/study_assets/data_schema/feature_columns.json"
  --fold-assignments "$FOLD_ROOT/fold_assignments.parquet"
  --fold-manifest "$FOLD_ROOT/fold_manifest.json"
  --task-file "$TASK"
  --preprocessing-root "$PREP_ROOT"
  --output-root "$OUT"
  --threshold-config "$THRESHOLD"
  --device cuda
)

check() {
  local f
  for f in "$CONFIG" "$ROOT/train_extracted.parquet" "$ROOT/validation_extracted.parquet" "$ROOT/test_extracted.parquet" \
           "$ROOT/study_assets/data_schema/feature_columns.json" "$FOLD_ROOT/fold_assignments.parquet" \
           "$FOLD_ROOT/fold_manifest.json" "$TASK" "$THRESHOLD"; do
    [[ -s "$f" ]] || die "Missing: $f"
  done
  for m in 0 1 2; do
    [[ -s "$PREP_ROOT/member_${m}/B/fold_preprocessing.json" ]] || die "Missing preprocessing member $m: $PREP_ROOT/member_${m}/B/fold_preprocessing.json"
  done
  echo "[OK] P3 Class-Balanced Hybrid; member=0; 30 epochs; buffer=4% per member (27778 slots)"
  echo "[OK] current=CB-Focal beta=0.9999 max_weight=4; replay=CE; distillation=off"
  echo "[OK] fold root: $FOLD_ROOT"
  echo "[OK] preprocessing root: $PREP_ROOT"
  echo "[OK] config: $CONFIG"
  echo "[OK] output: $OUT"
}

start() {
  [[ "${1:-}" == "0" ]] || die "Usage: $0 start 0"
  check
  [[ ! -e "$OUT/member_0" ]] || die "Output already exists: $OUT/member_0; set CB_HYB_OUT for a fresh run or resume intentionally."
  mkdir -p "$MONITOR"
  local stamp="member_0_$(date -u +%Y%m%dT%H%M%SZ)"
  local log="$MONITOR/$stamp"
  mkdir "$log"
  nohup env CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYTHON_BIN" src/training/fold_ensemble3_full.py "${COMMON[@]}" --member-id 0 --execute \
    > "$log/nohup.log" 2>&1 < /dev/null &
  echo $! > "$log/job.pid"
  printf '%s\n' "$log" > "$MONITOR/latest.txt"
  echo "[STARTED] member=0 PID=$(<"$log/job.pid")"
  echo "[LOG] $log/nohup.log"
}

status() {
  [[ -s "$MONITOR/latest.txt" ]] || die "No pilot has been started"
  local log="$(<"$MONITOR/latest.txt")"
  local pid="$log/job.pid"
  echo "LOG=$log"
  if kill -0 "$(<"$pid")" 2>/dev/null; then echo "[RUNNING] PID=$(<"$pid")"; else echo "[STOPPED/FINISHED]"; fi
  tail -n 80 "$log/nohup.log"
}

case "${1:-}" in
  check) check ;;
  dry-run) check; "$PYTHON_BIN" src/training/fold_ensemble3_full.py "${COMMON[@]}" --member-id 0 ;;
  start) start "${2:-}" ;;
  status) status ;;
  follow) [[ -s "$MONITOR/latest.txt" ]] || die "No pilot has been started"; tail -f "$(<"$MONITOR/latest.txt")/nohup.log" ;;
  *) echo "Usage: $0 check | dry-run | start 0 | status | follow"; exit 2 ;;
esac
