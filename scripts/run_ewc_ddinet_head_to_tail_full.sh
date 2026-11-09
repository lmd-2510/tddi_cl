#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${CIL_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
METHOD="${CIL_DDINET_FULL_RUN_METHOD:-ewc}"
case "$METHOD" in
  ewc) METHOD_LABEL="EWC" ;;
  agem) METHOD_LABEL="A-GEM" ;;
  *) echo "[STOP] Unsupported DDINet full-run method: $METHOD" >&2; exit 2 ;;
esac
CONFIG="$PROJECT_ROOT/configs/ordered_ddinet_ewc.json"
TRAIN="$PROJECT_ROOT/data/train_extracted.parquet"
VALIDATION="$PROJECT_ROOT/data/validation_extracted.parquet"
TEST="$PROJECT_ROOT/data/test_extracted.parquet"
FEATURE_COLUMNS="$PROJECT_ROOT/study_assets/data_schema/feature_columns.json"
TASK_FILE="$PROJECT_ROOT/study_assets/task_protocols/head_to_tail_tasks.json"
FOLD_ASSIGNMENTS="$PROJECT_ROOT/study_assets/stratified_3fold_seed42/fold_assignments.parquet"
FOLD_MANIFEST="$PROJECT_ROOT/study_assets/stratified_3fold_seed42/fold_manifest.json"
PREPROCESSING_ROOT="$PROJECT_ROOT/study_assets/preprocessing_ddinet_seed0_fold42"
SOURCE_SHARD_CACHE_ROOT="$PROJECT_ROOT/study_assets/source_shard_cache_ordered_seed0_fold42"
FINGERPRINT_CACHE_ROOT="$PROJECT_ROOT/study_assets/ddinet_fingerprint_cache_morgan1024"
MONITOR_ROOT="$PROJECT_ROOT/outputs/${METHOD}_ddinet_head_to_tail_full_monitor"
LATEST_FILE="$MONITOR_ROOT/latest.txt"

export CIL_DDINET_FINGERPRINT_CACHE="$FINGERPRINT_CACHE_ROOT"

die() { echo "[STOP] $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || die "Missing file: $1"; }
latest_run_home() { require_file "$LATEST_FILE"; local x; x="$(<"$LATEST_FILE")"; [[ -d "$x" ]] || die "Missing run home: $x"; printf '%s\n' "$x"; }
pid_is_running() { [[ -f "$1" ]] || return 1; local p; p="$(<"$1")"; [[ "$p" =~ ^[0-9]+$ ]] && kill -0 "$p" 2>/dev/null; }

prepare_assets() {
  cd "$PROJECT_ROOT"
  for path in "$PYTHON" "$TRAIN" "$VALIDATION" "$TEST" "$FEATURE_COLUMNS" "$TASK_FILE" "$FOLD_ASSIGNMENTS" "$FOLD_MANIFEST" "$SOURCE_SHARD_CACHE_ROOT/manifest.json"; do require_file "$path"; done
  if [[ ! -f "$FINGERPRINT_CACHE_ROOT/manifest.json" ]]; then
    echo "[PREPARE] Building label-free DDINet Morgan fingerprint cache."
    "$PYTHON" scripts/build_ddinet_fingerprint_cache.py \
      --train "$TRAIN" --validation "$VALIDATION" --test "$TEST" \
      --outdir "$FINGERPRINT_CACHE_ROOT"
  else
    echo "[PREPARE] DDINet fingerprint cache already exists: $FINGERPRINT_CACHE_ROOT"
  fi
  for member in 0 1 2; do
    local out="$PREPROCESSING_ROOT/member_${member}"
    if [[ ! -f "$out/fold_preprocessing.json" ]]; then
      echo "[PREPARE] Writing raw-identity fold artifact for member $member."
      "$PYTHON" scripts/prepare_fold_preprocessing.py \
        --assignments "$FOLD_ASSIGNMENTS" --manifest "$FOLD_MANIFEST" \
        --train "$TRAIN" --validation "$VALIDATION" --test "$TEST" \
        --task-file "$TASK_FILE" --feature-cols "$FEATURE_COLUMNS" \
        --member-id "$member" --validation-fold "$member" --experiment-seed 0 \
        --fold-seed 42 --policy raw_identity --outdir "$out"
    else
      echo "[PREPARE] Raw-identity artifact already exists: $out/fold_preprocessing.json"
    fi
  done
}

full_command() {
  local output_root="$1"; local execute_flag="${2:-}"
  local command=("$PYTHON" "$PROJECT_ROOT/src/training/fold_ensemble3_full.py" \
    --config "$CONFIG" --method "$METHOD" --train "$TRAIN" --validation "$VALIDATION" --test "$TEST" \
    --feature-cols "$FEATURE_COLUMNS" --fold-assignments "$FOLD_ASSIGNMENTS" \
    --fold-manifest "$FOLD_MANIFEST" --task-file "$TASK_FILE" \
    --preprocessing-root "$PREPROCESSING_ROOT" --output-root "$output_root" \
    --device cuda --python "$PYTHON")
  [[ "$execute_flag" == execute ]] && command+=(--execute)
  printf '%q ' "${command[@]}"; printf '\n'
}

check_all() {
  cd "$PROJECT_ROOT"
  prepare_assets
  "$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available(): raise SystemExit('[STOP] CUDA unavailable')
free,total=torch.cuda.mem_get_info()
print(f'[OK] CUDA: {torch.cuda.get_device_name(0)}; free={free/1024**3:.2f} GiB; total={total/1024**3:.2f} GiB')
PY
  for member in 0 1 2; do require_file "$PREPROCESSING_ROOT/member_${member}/fold_preprocessing.json"; done
  require_file "$FINGERPRINT_CACHE_ROOT/manifest.json"
  echo "[CHECK] Validating DDINet $METHOD_LABEL full-run plan; no model training."
  export CIL_SOURCE_SHARD_CACHE_ROOT="$SOURCE_SHARD_CACHE_ROOT"
  local dry_root="$PROJECT_ROOT/outputs/_${METHOD}_ddinet_head_to_tail_dry_run_only"
  eval "$(full_command "$dry_root")"
  echo "[OK] DDINet + $METHOD_LABEL + fixed 178-head + P1 head-to-tail plan passed."
}

postprocess() {
  local output_root="$1"
  "$PYTHON" "$PROJECT_ROOT/src/eval/report.py" --full-root "$output_root" --task-file "$TASK_FILE" --overwrite
  "$PYTHON" "$PROJECT_ROOT/scripts/visualize_cil_run.py" --run-root "$output_root" \
    --outdir "$output_root/visualizations_ensemble" --task-id 7 --member-id 0 --member-ids 0 1 2 \
    --ensemble --split test --task-file "$TASK_FILE" --train "$TRAIN" --validation "$VALIDATION" --test "$TEST"
  local archive="$output_root/${output_root##*/}_metrics_review.zip"
  "$PYTHON" -m src.eval.package --run-root "$output_root" --archive "$archive" --overwrite
}

worker() {
  local run_home="$1" output_root="$2"
  mkdir -p "$run_home" "$output_root"
  exec > >(tee -a "$run_home/main.log") 2>&1
  set +e
  echo "[RECIPE] $METHOD_LABEL + DDINet + Morgan1024 ordered pair + fixed 178 head + P1 head-to-tail"
  export CIL_SOURCE_SHARD_CACHE_ROOT="$SOURCE_SHARD_CACHE_ROOT"
  eval "$(full_command "$output_root" execute)"
  local code=$?
  if [[ $code -eq 0 ]]; then
    postprocess "$output_root"
    code=$?
  fi
  printf '%s\n' "$code" > "$run_home/exit_code.txt"
  printf '%s\n' "$code" > "$output_root/runner_exit_code.txt"
  [[ $code -eq 0 ]] && echo "[COMPLETED] Review ZIP: $output_root/${output_root##*/}_metrics_review.zip" || echo "[FAILED] exit_code=$code"
  exit "$code"
}

start_new() {
  cd "$PROJECT_ROOT"; check_all; mkdir -p "$MONITOR_ROOT"
  if [[ -f "$LATEST_FILE" ]]; then local old; old="$(<"$LATEST_FILE")"; pid_is_running "$old/job.pid" && die "A run is already active."; fi
  local id run_home output_root; id="$(date -u +%Y%m%dT%H%M%SZ)"
  run_home="$MONITOR_ROOT/run_${id}"; output_root="$PROJECT_ROOT/outputs/${METHOD}_ddinet_head_to_tail_fixed_head_full_seed0_${id}"
  mkdir -p "$run_home" "$output_root"; printf '%s\n' "$run_home" > "$LATEST_FILE"; printf '%s\n' "$output_root" > "$run_home/output_root.txt"
  nohup bash "$0" _worker "$run_home" "$output_root" > "$run_home/nohup.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$run_home/job.pid"; echo "[STARTED] PID=$(<"$run_home/job.pid")"; echo "[LOG] $run_home/main.log"
}

status() {
  local run_home output_root; run_home="$(latest_run_home)"; output_root="$(<"$run_home/output_root.txt")"
  if pid_is_running "$run_home/job.pid"; then echo "[RUNNING] PID=$(<"$run_home/job.pid")"; elif [[ -f "$run_home/exit_code.txt" ]]; then echo "[FINISHED] exit_code=$(<"$run_home/exit_code.txt")"; else echo "[NOT RUNNING]"; fi
  for member in 0 1 2; do local ck=""; [[ -d "$output_root/member_${member}/checkpoints" ]] && ck="$(find "$output_root/member_${member}/checkpoints" -maxdepth 1 -name 'task_*.pt' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1)"; echo "M${member}: ${ck:-not started}"; done
  [[ -f "$output_root/full_manifest.json" ]] && echo "Offline ensemble: completed" || echo "Offline ensemble: pending"
}

resume_latest() {
  cd "$PROJECT_ROOT"
  local run_home output_root
  run_home="$(latest_run_home)"
  require_file "$run_home/output_root.txt"
  output_root="$(<"$run_home/output_root.txt")"
  if pid_is_running "$run_home/job.pid"; then
    die "The latest launcher is still running."
  fi
  check_all
  [[ ! -f "$run_home/exit_code.txt" ]] || mv -f "$run_home/exit_code.txt" "$run_home/previous_exit_code.txt"
  nohup bash "$0" _worker "$run_home" "$output_root" > "$run_home/nohup.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$run_home/job.pid"
  echo "[RESUMED] PID=$(<"$run_home/job.pid")"
  echo "[OUTPUT] $output_root"
  echo "[LOG] $run_home/main.log"
}

case "${1:-}" in
  prepare) prepare_assets ;;
  check) check_all ;;
  start) start_new ;;
  resume) resume_latest ;;
  status) status ;;
  follow) tail -n 100 -F "$(latest_run_home)/main.log" ;;
  _worker) worker "$2" "$3" ;;
  *) echo "Usage: $0 prepare|check|start|resume|status|follow"; exit 2 ;;
esac
