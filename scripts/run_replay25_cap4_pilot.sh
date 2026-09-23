#!/usr/bin/env bash
set -euo pipefail

# One-member pilot: baseline hybrid/no-distillation with replay fraction 25%
# and repeat cap 4. The other 2 members are intentionally not started.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${REPLAY25_PYTHON:-python}"
GPU="${REPLAY25_GPU_ID:-0}"
CONFIG="${REPLAY25_CONFIG:-$ROOT/configs/p3_hybrid_replay25_cap4_pilot.json}"
OUT="${REPLAY25_OUT:-$ROOT/outputs/p3_hybrid_replay25_cap4_pilot_seed0}"
MONITOR="${REPLAY25_MONITOR:-$ROOT/outputs/replay25_cap4_pilot_monitor}"
TASK="$ROOT/study_assets/task_protocols/tail_to_head_tasks.json"
THRESHOLD="$ROOT/configs/eval_tddi_p3_ensemble_entropy_threshold.json"

die() { echo "[STOP] $*" >&2; exit 1; }

find_fold_root() {
  if [[ -n "${REPLAY25_FOLD_ROOT:-}" ]]; then printf '%s\n' "$REPLAY25_FOLD_ROOT"; return; fi
  local preferred="$ROOT/study_assets/stratified_3fold_seed42"
  if [[ -s "$preferred/fold_assignments.parquet" && -s "$preferred/fold_manifest.json" ]]; then
    printf '%s\n' "$preferred"; return
  fi
  mapfile -t found < <(find "$ROOT/outputs" -type f -name fold_assignments.parquet -printf '%h\n' 2>/dev/null | sort -u)
  (( ${#found[@]} == 1 )) || die "Set REPLAY25_FOLD_ROOT to the directory containing fold_assignments.parquet and fold_manifest.json"
  printf '%s\n' "${found[0]}"
}

FOLD_ROOT="$(find_fold_root)"
PREP_ROOT="${REPLAY25_PREP_ROOT:-$ROOT/study_assets/preprocessing_p3_seed0_fold42}"
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
    [[ -s "$PREP_ROOT/member_${m}/B/fold_preprocessing.json" ]] || die "Missing preprocessing member $m"
  done
  echo "[OK] config: $CONFIG"
  echo "[OK] fold root: $FOLD_ROOT"
  echo "[OK] preprocessing root: $PREP_ROOT"
  echo "[OK] member budget: 27778; replay fraction: 25%; repeat cap: 4"
  echo "[OK] output: $OUT"
}

start() {
  [[ "${1:-}" == "0" ]] || die "Usage: $0 start 0"
  check
  [[ ! -e "$OUT/member_0" ]] || die "Output already exists: $OUT/member_0; use a new REPLAY25_OUT or resume intentionally."
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
  if kill -0 "$(<"$pid")" 2>/dev/null; then echo "[RUNNING] PID=$(<$pid)"; else echo "[STOPPED/FINISHED]"; fi
  tail -n 60 "$log/nohup.log"
}

case "${1:-}" in
  check) check ;;
  dry-run) check; "$PYTHON_BIN" src/training/fold_ensemble3_full.py "${COMMON[@]}" --member-id 0 ;;
  start) start "${2:-}" ;;
  status) status ;;
  follow) [[ -s "$MONITOR/latest.txt" ]] || die "No pilot has been started"; tail -f "$(<"$MONITOR/latest.txt")/nohup.log" ;;
  *) echo "Usage: $0 check | dry-run | start 0 | status | follow"; exit 2 ;;
esac
