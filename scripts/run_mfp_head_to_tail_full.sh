#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$PROJECT_ROOT/scripts/run_mfp_head_to_tail_full.sh"
PYTHON="${CIL_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
METHOD="${1:-}"
ACTION="${2:-check}"
case "$METHOD" in
  er) METHOD_LABEL="ER" ;;
  agem) METHOD_LABEL="A-GEM" ;;
  ewc) METHOD_LABEL="EWC" ;;
  *) echo "Usage: $0 {er|agem|ewc} {prepare|check|start|resume|status|follow|package}"; exit 2 ;;
esac
CONFIG="$PROJECT_ROOT/configs/ordered_mfp_${METHOD}.json"
TRAIN="$PROJECT_ROOT/data/train_extracted.parquet"
VALIDATION="$PROJECT_ROOT/data/validation_extracted.parquet"
TEST="$PROJECT_ROOT/data/test_extracted.parquet"
FEATURE_COLUMNS="$PROJECT_ROOT/study_assets/data_schema/feature_columns.json"
TASK_FILE="$PROJECT_ROOT/study_assets/task_protocols/head_to_tail_tasks.json"
FOLD_ROOT="$PROJECT_ROOT/study_assets/stratified_3fold_seed42"
PREPROCESSING_ROOT="$PROJECT_ROOT/study_assets/preprocessing_mfp_seed0_fold42"
SOURCE_CACHE="$PROJECT_ROOT/study_assets/source_shard_cache_ordered_seed0_fold42"
FINGERPRINT_CACHE="$PROJECT_ROOT/study_assets/mfp_fingerprint_cache_morgan2048"
MONITOR_ROOT="$PROJECT_ROOT/outputs/${METHOD}_mfp_head_to_tail_full_monitor"
export CIL_MFP_FINGERPRINT_CACHE="$FINGERPRINT_CACHE"
export CIL_SOURCE_SHARD_CACHE_ROOT="$SOURCE_CACHE"

die() { echo "[STOP] $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || die "Missing file: $1"; }
latest_home() {
  require_file "$MONITOR_ROOT/latest.txt"
  local run_home
  run_home="$(<"$MONITOR_ROOT/latest.txt")"
  [[ -d "$run_home" ]] || die "Missing monitor directory: $run_home"
  printf '%s\n' "$run_home"
}
running() {
  [[ -f "$1/job.pid" ]] || return 1
  local job_pid
  job_pid="$(<"$1/job.pid")"
  [[ "$job_pid" =~ ^[0-9]+$ ]] && kill -0 "$job_pid" 2>/dev/null
}

prepare_assets() {
  cd "$PROJECT_ROOT"
  for path in "$PYTHON" "$CONFIG" "$TRAIN" "$VALIDATION" "$TEST" "$FEATURE_COLUMNS" "$TASK_FILE" \
    "$FOLD_ROOT/fold_assignments.parquet" "$FOLD_ROOT/fold_manifest.json" "$SOURCE_CACHE/manifest.json"; do
    require_file "$path"
  done
  if [[ ! -f "$FINGERPRINT_CACHE/manifest.json" ]]; then
    "$PYTHON" scripts/build_mfp_fingerprint_cache.py \
      --train "$TRAIN" --validation "$VALIDATION" --test "$TEST" --outdir "$FINGERPRINT_CACHE"
  fi
  for member in 0 1 2; do
    if [[ ! -f "$PREPROCESSING_ROOT/member_${member}/fold_preprocessing.json" ]]; then
      "$PYTHON" scripts/prepare_fold_preprocessing.py \
        --assignments "$FOLD_ROOT/fold_assignments.parquet" --manifest "$FOLD_ROOT/fold_manifest.json" \
        --train "$TRAIN" --validation "$VALIDATION" --test "$TEST" \
        --task-file "$TASK_FILE" --feature-cols "$FEATURE_COLUMNS" \
        --member-id "$member" --validation-fold "$member" --experiment-seed 0 \
        --fold-seed 42 --policy raw_identity --outdir "$PREPROCESSING_ROOT/member_${member}"
    fi
  done
}

full_command() {
  local output="$1"
  shift
  "$PYTHON" "$PROJECT_ROOT/src/training/fold_ensemble3_full.py" \
    --config "$CONFIG" --method "$METHOD" --train "$TRAIN" --validation "$VALIDATION" --test "$TEST" \
    --feature-cols "$FEATURE_COLUMNS" --task-file "$TASK_FILE" \
    --fold-assignments "$FOLD_ROOT/fold_assignments.parquet" --fold-manifest "$FOLD_ROOT/fold_manifest.json" \
    --preprocessing-root "$PREPROCESSING_ROOT" --output-root "$output" --device cuda --python "$PYTHON" "$@"
}

check() {
  prepare_assets
  "$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("[STOP] CUDA unavailable")
print(f"[OK] CUDA {torch.cuda.get_device_name(0)}", flush=True)
PY
  full_command "$PROJECT_ROOT/outputs/_${METHOD}_mfp_head_to_tail_dry_run_only"
}

postprocess() {
  local output="$1"
  "$PYTHON" "$PROJECT_ROOT/src/eval/report.py" --full-root "$output" --task-file "$TASK_FILE" --overwrite || return $?
  "$PYTHON" "$PROJECT_ROOT/scripts/visualize_cil_run.py" --run-root "$output" \
    --outdir "$output/visualizations_ensemble" --task-id 7 --member-id 0 --member-ids 0 1 2 \
    --ensemble --split test --task-file "$TASK_FILE" --train "$TRAIN" --validation "$VALIDATION" --test "$TEST" || return $?
  "$PYTHON" -m src.eval.package --run-root "$output" --archive "$output/${output##*/}_metrics_review.zip" --overwrite
}

worker() {
  local run_home="$1" output="$2"
  cd "$PROJECT_ROOT"
  exec > >(tee -a "$run_home/main.log") 2>&1
  set +e
  echo "[RECIPE] $METHOD_LABEL + MFP-Siamese + Morgan2048/drug + fixed 178 head + P1 head-to-tail + three members"
  echo "[TRAINING] Adam lr=0.0001 weight_decay=0 batch=512 effective_batch=512 epochs<=200 patience=5; validation Macro-F1 selection"
  if [[ "$METHOD" == agem ]]; then
    echo "[MEMORY] A-GEM task-uniform seeded memory, budget 27778/member, reference batch 256, gradient projection"
  elif [[ "$METHOD" == ewc ]]; then
    echo "[REGULARIZATION] CE + cumulative EWC lambda=1000; Fisher and parameter anchors; exemplar slots=0"
  else
    echo "[MEMORY] Uniform reservoir 27778/member; epoch current/replay draws 1:1"
  fi
  echo "[MODEL] Singleton BatchNorm uses running statistics"
  full_command "$output" --execute
  local code=$?
  if [[ $code -eq 0 ]]; then
    postprocess "$output"
    code=$?
  fi
  printf '%s\n' "$code" > "$run_home/exit_code.txt"
  printf '%s\n' "$code" > "$output/runner_exit_code.txt"
  if [[ $code -eq 0 ]]; then
    echo "[COMPLETED] $output/${output##*/}_metrics_review.zip"
  else
    echo "[FAILED] exit_code=$code; inspect $run_home/main.log"
  fi
  exit "$code"
}

launch() {
  local action="$1" run_home output
  cd "$PROJECT_ROOT"
  if [[ -f "$MONITOR_ROOT/latest.txt" ]]; then
    run_home="$(latest_home)"
    if running "$run_home"; then die "A $METHOD_LABEL-MFP launcher is already running."; fi
  fi
  check
  if [[ "$action" == resume ]]; then
    run_home="$(latest_home)"
    require_file "$run_home/output_root.txt"
    output="$(<"$run_home/output_root.txt")"
    [[ ! -f "$run_home/exit_code.txt" ]] || mv -f "$run_home/exit_code.txt" "$run_home/previous_exit_code.txt"
  else
    local run_id
    run_id="$(date -u +%Y%m%dT%H%M%SZ)"
    run_home="$MONITOR_ROOT/run_${run_id}"
    output="$PROJECT_ROOT/outputs/${METHOD}_mfp_head_to_tail_fixed_head_full_seed0_${run_id}"
    mkdir -p "$run_home" "$output"
    printf '%s\n' "$run_home" > "$MONITOR_ROOT/latest.txt"
    printf '%s\n' "$output" > "$run_home/output_root.txt"
  fi
  nohup bash "$SCRIPT_PATH" "$METHOD" _worker "$run_home" "$output" > "$run_home/nohup.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$run_home/job.pid"
  echo "[STARTED] PID=$(<"$run_home/job.pid")"
  echo "[OUTPUT] $output"
  echo "[LOG] $run_home/main.log"
}

status() {
  local run_home output
  run_home="$(latest_home)"
  output="$(<"$run_home/output_root.txt")"
  if running "$run_home"; then
    echo "[RUNNING] PID=$(<"$run_home/job.pid")"
  elif [[ -f "$run_home/exit_code.txt" ]]; then
    echo "[FINISHED] exit_code=$(<"$run_home/exit_code.txt")"
  else
    echo "[NOT RUNNING]"
  fi
  echo "[OUTPUT] $output"
  for member in 0 1 2; do
    if [[ -f "$output/member_${member}/checkpoints/task_7.pt" ]]; then
      echo "M${member}: task 7 checkpoint ready"
    else
      echo "M${member}: pending or in progress"
    fi
  done
  [[ ! -f "$output/${output##*/}_metrics_review.zip" ]] || echo "[REVIEW] ZIP ready"
}

case "$ACTION" in
  prepare) prepare_assets ;;
  check) check ;;
  start|resume) launch "$ACTION" ;;
  status) status ;;
  follow) tail -n 100 -F "$(latest_home)/main.log" ;;
  package)
    run_home="$(latest_home)"
    if running "$run_home"; then die "Wait for the active run to finish before packaging."; fi
    cd "$PROJECT_ROOT"
    postprocess "$(<"$run_home/output_root.txt")"
    ;;
  _worker) worker "$3" "$4" ;;
  *) echo "Usage: $0 {er|agem|ewc} {prepare|check|start|resume|status|follow|package}"; exit 2 ;;
esac
