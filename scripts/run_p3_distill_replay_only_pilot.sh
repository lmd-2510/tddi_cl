#!/usr/bin/env bash
set -euo pipefail

# One-member full P3 pilot, holding the best baseline settings fixed and
# applying logit/feature distillation only to replay examples.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${DISTILL_REPLAY_PYTHON:-python}"
GPU="${DISTILL_REPLAY_GPU_ID:-0}"
CONFIG="${DISTILL_REPLAY_CONFIG:-$ROOT/configs/p3_hybrid_distill_replay_only_pilot.json}"
OUT="${DISTILL_REPLAY_OUT:-$ROOT/outputs/p3_hybrid_distill_replay_only_pilot_seed0}"
MONITOR="${DISTILL_REPLAY_MONITOR:-$ROOT/outputs/p3_distill_replay_only_pilot_monitor}"
TASK="$ROOT/study_assets/task_protocols/tail_to_head_tasks.json"
THRESHOLD="$ROOT/configs/eval_tddi_p3_ensemble_entropy_threshold.json"

die() { echo "[STOP] $*" >&2; exit 1; }

find_fold_root() {
  if [[ -n "${DISTILL_REPLAY_FOLD_ROOT:-}" ]]; then printf '%s\n' "$DISTILL_REPLAY_FOLD_ROOT"; return; fi
  local preferred="$ROOT/study_assets/stratified_3fold_seed42"
  if [[ -s "$preferred/fold_assignments.parquet" && -s "$preferred/fold_manifest.json" ]]; then
    printf '%s\n' "$preferred"; return
  fi
  die "Set DISTILL_REPLAY_FOLD_ROOT to the directory containing fold_assignments.parquet and fold_manifest.json"
}

FOLD_ROOT="$(find_fold_root)"
PREP_ROOT="${DISTILL_REPLAY_PREP_ROOT:-$ROOT/study_assets/preprocessing_p3_seed0_fold42}"
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
           "$FOLD_ROOT/fold_manifest.json" "$TASK" "$THRESHOLD" \
           "$PREP_ROOT/member_0/B/fold_preprocessing.json"; do
    [[ -s "$f" ]] || die "Missing: $f"
  done
  [[ "$(sha256sum "$TASK" | awk '{print $1}')" == "0d64c465b0c4bd34f66e6c76088b6b73fd60839ade3e56017b5fd36c21a26e79" ]] \
    || die "P3 task hash mismatch"
  echo "[OK] baseline settings: P3, 30 epochs, 27778 buffer slots/member, replay=12.5%, cap=3"
  echo "[OK] loss: hybrid_distill_replay_only (KL + feature MSE on replay samples only)"
  echo "[OK] member: 0 only; output: $OUT"
}

start() {
  check
  [[ ! -e "$OUT/member_0" ]] || die "Output already exists: $OUT/member_0; choose a new DISTILL_REPLAY_OUT to start a fresh pilot."
  mkdir -p "$MONITOR"
  local stamp="member_0_$(date -u +%Y%m%dT%H%M%SZ)"
  local log="$MONITOR/$stamp"
  mkdir "$log"
  nohup bash -c '
    set -euo pipefail
    python_bin="$1"
    gpu="$2"
    run_root="$3"
    shift 3
    env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$python_bin" "$@"
    "$python_bin" scripts/visualize_cil_run.py \
      --run-root "$run_root" --outdir "$run_root/visualizations/member_0_validation" \
      --member-id 0 --member-ids 0 --split validation
  ' _ "$PYTHON_BIN" "$GPU" "$OUT" \
    src/training/fold_ensemble3_full.py "${COMMON[@]}" --member-id 0 --execute \
    > "$log/nohup.log" 2>&1 < /dev/null &
  echo $! > "$log/job.pid"
  printf '%s\n' "$log" > "$MONITOR/latest.txt"
  echo "[STARTED] member=0 PID=$(<"$log/job.pid")"
  echo "[LOG] $log/nohup.log (sau train thành công sẽ tự tạo visualization validation)"
}

status() {
  [[ -s "$MONITOR/latest.txt" ]] || die "No pilot has been started"
  local log
  log="$(<"$MONITOR/latest.txt")"
  if kill -0 "$(<"$log/job.pid")" 2>/dev/null; then echo "[RUNNING] PID=$(<"$log/job.pid")"; else echo "[STOPPED/FINISHED]"; fi
  tail -n 60 "$log/nohup.log"
}

case "${1:-}" in
  check) check ;;
  dry-run) check; "$PYTHON_BIN" src/training/fold_ensemble3_full.py "${COMMON[@]}" --member-id 0 ;;
  start) start ;;
  status) status ;;
  follow) [[ -s "$MONITOR/latest.txt" ]] || die "No pilot has been started"; tail -f "$(<"$MONITOR/latest.txt")/nohup.log" ;;
  *) echo "Usage: $0 check | dry-run | start | status | follow"; exit 2 ;;
esac
