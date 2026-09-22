#!/usr/bin/env bash
set -euo pipefail

# One-GPU overnight ablation: first Hybrid+logit/feature distillation, then
# the identical Hybrid run with both distillation losses disabled.
# Each study itself runs member 0 -> 1 -> 2 and then its offline ensemble/UE.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${P3AB_PYTHON:-python}"
GPU="${P3AB_GPU_ID:-0}"
DISTILL_CONFIG="${P3AB_DISTILL_CONFIG:-$ROOT/configs/p3_hybrid_distill_full8_e30_mem4.json}"
HYBRID_CONFIG="${P3AB_HYBRID_CONFIG:-$ROOT/configs/p3_hybrid_full8_e30_mem4.json}"
DISTILL_OUT="${P3AB_DISTILL_OUT:-$ROOT/outputs/p3_hybrid_distill_full8_e30_mem4_seed0}"
HYBRID_OUT="${P3AB_HYBRID_OUT:-$ROOT/outputs/p3_hybrid_full8_e30_mem4_seed0}"
MONITOR="${P3AB_MONITOR:-$ROOT/outputs/p3_loss_ab_e30_mem4_monitor}"
TASK="$ROOT/study_assets/task_protocols/tail_to_head_tasks.json"
THRESHOLD="$ROOT/configs/eval_tddi_p3_ensemble_entropy_balanced_accuracy_threshold.json"
die() { echo "[STOP] $*" >&2; exit 1; }

find_fold_root() {
  if [[ -n "${P3AB_FOLD_ROOT:-}" ]]; then printf '%s\n' "$P3AB_FOLD_ROOT"; return; fi
  local preferred="$ROOT/study_assets/stratified_3fold_seed42"
  if [[ -s "$preferred/fold_assignments.parquet" && -s "$preferred/fold_manifest.json" ]]; then printf '%s\n' "$preferred"; return; fi
  mapfile -t found < <(find "$ROOT/outputs" -type f -name fold_assignments.parquet -printf '%h\n' 2>/dev/null | sort -u)
  (( ${#found[@]} == 1 )) || die "Set P3AB_FOLD_ROOT to the directory with fold_assignments.parquet and fold_manifest.json"
  printf '%s\n' "${found[0]}"
}

find_prep_root() {
  if [[ -n "${P3AB_PREP_ROOT:-}" ]]; then printf '%s\n' "$P3AB_PREP_ROOT"; return; fi
  local preferred="$ROOT/study_assets/preprocessing_p3_seed0_fold42"
  if [[ -s "$preferred/member_0/B/fold_preprocessing.json" && -s "$preferred/member_1/B/fold_preprocessing.json" && -s "$preferred/member_2/B/fold_preprocessing.json" ]]; then printf '%s\n' "$preferred"; return; fi
  die "Set P3AB_PREP_ROOT to the root containing member_0/1/2/B/fold_preprocessing.json"
}

FOLD_ROOT="$(find_fold_root)"
PREP_ROOT="$(find_prep_root)"
COMMON=(
  --train "$ROOT/train_extracted.parquet" --validation "$ROOT/validation_extracted.parquet" --test "$ROOT/test_extracted.parquet"
  --feature-cols "$ROOT/study_assets/data_schema/feature_columns.json"
  --fold-assignments "$FOLD_ROOT/fold_assignments.parquet" --fold-manifest "$FOLD_ROOT/fold_manifest.json"
  --task-file "$TASK" --preprocessing-root "$PREP_ROOT" --threshold-config "$THRESHOLD" --device cuda
  --member-id 0 --member-id 1 --member-id 2
)

check() {
  local f prep_member
  for f in "$DISTILL_CONFIG" "$HYBRID_CONFIG" "$ROOT/train_extracted.parquet" "$ROOT/validation_extracted.parquet" "$ROOT/test_extracted.parquet" "$ROOT/study_assets/data_schema/feature_columns.json" "$FOLD_ROOT/fold_assignments.parquet" "$FOLD_ROOT/fold_manifest.json" "$TASK" "$THRESHOLD"; do [[ -s "$f" ]] || die "Missing: $f"; done
  for prep_member in 0 1 2; do [[ -s "$PREP_ROOT/member_${prep_member}/B/fold_preprocessing.json" ]] || die "Missing preprocessing member $prep_member"; done
  [[ "$(sha256sum "$TASK" | awk '{print $1}')" == "0d64c465b0c4bd34f66e6c76088b6b73fd60839ade3e56017b5fd36c21a26e79" ]] || die "P3 task hash mismatch"
  echo "[OK] A: Hybrid+distill, 30 epochs, 27778 buffer/member -> $DISTILL_OUT"
  echo "[OK] B: Hybrid (no logit/feature distill), same settings -> $HYBRID_OUT"
}

study() {
  local label="$1" config="$2" out="$3"
  echo "=== BEGIN $label: $(date -u +%FT%TZ) ==="
  "$PYTHON_BIN" src/training/fold_ensemble3_full.py --config "$config" "${COMMON[@]}" --output-root "$out" --execute
  echo "=== COMPLETE $label: $(date -u +%FT%TZ) ==="
}

dry_run() {
  check
  "$PYTHON_BIN" src/training/fold_ensemble3_full.py --config "$DISTILL_CONFIG" "${COMMON[@]}" --output-root "$DISTILL_OUT"
  "$PYTHON_BIN" src/training/fold_ensemble3_full.py --config "$HYBRID_CONFIG" "${COMMON[@]}" --output-root "$HYBRID_OUT"
}

run() { check; study distill "$DISTILL_CONFIG" "$DISTILL_OUT"; study no_distill "$HYBRID_CONFIG" "$HYBRID_OUT"; }

start() {
  check
  mkdir -p "$MONITOR"
  local log="$MONITOR/run_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir "$log"
  nohup env CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash "$0" run > "$log/nohup.log" 2>&1 < /dev/null &
  echo $! > "$log/job.pid"
  printf '%s\n' "$log" > "$MONITOR/latest.txt"
  echo "[STARTED] PID=$(<"$log/job.pid")"
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
  run) run ;;
  start) start ;;
  status) status ;;
  follow) [[ -s "$MONITOR/latest.txt" ]] || die "No run has been started"; tail -f "$(<"$MONITOR/latest.txt")/nohup.log" ;;
  *) echo "Usage: $0 check | dry-run | start | status | follow"; exit 2 ;;
esac
