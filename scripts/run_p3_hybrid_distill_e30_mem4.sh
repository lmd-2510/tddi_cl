#!/usr/bin/env bash
set -euo pipefail

# P3 Hybrid+distillation, full 8 tasks, 30 epochs, 4% buffer per member.
# Members run sequentially: start 0, wait for task 7, then start 1, then 2.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${P3D30_PYTHON:-python}"
GPU="${P3D30_GPU_ID:-0}"
CONFIG="${P3D30_CONFIG:-$ROOT/configs/p3_hybrid_distill_full8_e30_mem4.json}"
OUT="${P3D30_OUT:-$ROOT/outputs/p3_hybrid_distill_full8_e30_mem4_seed0}"
MONITOR="${P3D30_MONITOR:-$ROOT/outputs/p3_hybrid_distill_e30_mem4_monitor}"
TASK="$ROOT/study_assets/task_protocols/tail_to_head_tasks.json"
THRESHOLD="$ROOT/configs/eval_tddi_p3_ensemble_entropy_balanced_accuracy_threshold.json"
die() { echo "[STOP] $*" >&2; exit 1; }

find_fold_root() {
  if [[ -n "${P3D30_FOLD_ROOT:-}" ]]; then printf '%s\n' "$P3D30_FOLD_ROOT"; return; fi
  local preferred="$ROOT/study_assets/stratified_3fold_seed42"
  if [[ -s "$preferred/fold_assignments.parquet" && -s "$preferred/fold_manifest.json" ]]; then
    printf '%s\n' "$preferred"; return
  fi
  mapfile -t found < <(find "$ROOT/outputs" -type f -name fold_assignments.parquet -printf '%h\n' 2>/dev/null | sort -u)
  (( ${#found[@]} == 1 )) || die "Set P3D30_FOLD_ROOT to the directory containing fold_assignments.parquet and fold_manifest.json"
  printf '%s\n' "${found[0]}"
}

find_prep_root() {
  if [[ -n "${P3D30_PREP_ROOT:-}" ]]; then printf '%s\n' "$P3D30_PREP_ROOT"; return; fi
  local preferred="$ROOT/study_assets/preprocessing_p3_seed0_fold42"
  if [[ -s "$preferred/member_0/B/fold_preprocessing.json" && -s "$preferred/member_1/B/fold_preprocessing.json" && -s "$preferred/member_2/B/fold_preprocessing.json" ]]; then
    printf '%s\n' "$preferred"; return
  fi
  mapfile -t found < <(find "$ROOT/study_assets" -type f -path '*/member_0/B/fold_preprocessing.json' -printf '%h\n' 2>/dev/null | sed 's#/member_0/B$##' | sort -u)
  (( ${#found[@]} == 1 )) || die "Set P3D30_PREP_ROOT to the preprocessing root containing member_0/1/2/B/fold_preprocessing.json"
  printf '%s\n' "${found[0]}"
}

FOLD_ROOT="$(find_fold_root)"
PREP_ROOT="$(find_prep_root)"
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
  for prep_member in 0 1 2; do
    [[ -s "$PREP_ROOT/member_${prep_member}/B/fold_preprocessing.json" ]] || die "Missing preprocessing member $prep_member"
  done
  [[ "$(sha256sum "$TASK" | awk '{print $1}')" == "0d64c465b0c4bd34f66e6c76088b6b73fd60839ade3e56017b5fd36c21a26e79" ]] || die "P3 task hash mismatch"
  echo "[OK] P3 Hybrid+distillation; epochs=30; buffer=4% per member (27778); total=83334"
  echo "[OK] config=$CONFIG"
  echo "[OK] output=$OUT"
}

dry_run() { check; "$PYTHON_BIN" src/training/fold_ensemble3_full.py "${COMMON[@]}" --member-id 0; }

start() {
  local member="${1:-}"
  [[ "$member" =~ ^[012]$ ]] || die "Usage: $0 start 0|1|2"
  check
  if [[ "$member" == 1 ]]; then
    [[ -s "$OUT/member_0/checkpoints/task_7.pt" && -s "$OUT/member_0/task_7/completed_task.json" ]] || die "Member 0 chưa hoàn tất task 7."
  elif [[ "$member" == 2 ]]; then
    [[ -s "$OUT/member_1/checkpoints/task_7.pt" && -s "$OUT/member_1/task_7/completed_task.json" ]] || die "Member 1 chưa hoàn tất task 7."
  fi
  mkdir -p "$MONITOR"
  local stamp="member_${member}_$(date -u +%Y%m%dT%H%M%SZ)" log="$MONITOR/$stamp"
  mkdir "$log"
  nohup env CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYTHON_BIN" src/training/fold_ensemble3_full.py "${COMMON[@]}" --member-id "$member" --execute \
    > "$log/nohup.log" 2>&1 < /dev/null &
  echo $! > "$log/job.pid"
  printf '%s\n' "$log" > "$MONITOR/latest.txt"
  echo "[STARTED] member=$member PID=$(<"$log/job.pid")"
  echo "[LOG] $log/nohup.log"
}

status() {
  [[ -s "$MONITOR/latest.txt" ]] || die "No run has been started"
  local log="$(<"$MONITOR/latest.txt")" pid="$log/job.pid"
  echo "LOG=$log"
  if kill -0 "$(<"$pid")" 2>/dev/null; then echo "[RUNNING] PID=$(<$pid)"; else echo "[STOPPED/FINISHED]"; fi
  tail -n 40 "$log/nohup.log"
}

case "${1:-}" in
  check) check ;;
  dry-run) dry_run ;;
  start) start "${2:-}" ;;
  status) status ;;
  follow) [[ -s "$MONITOR/latest.txt" ]] || die "No run has been started"; tail -f "$(<"$MONITOR/latest.txt")/nohup.log" ;;
  *) echo "Usage: $0 check | dry-run | start 0|1|2 | status | follow"; exit 2 ;;
esac
