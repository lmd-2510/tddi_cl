#!/usr/bin/env bash
set -euo pipefail

# Full model-merging run for Task Arithmetic, TIES, OPCM, or MagMax.
# Usage: bash scripts/run_model_merging_full.sh {task_arithmetic|ties|opcm|magmax} {check|start|status|follow}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
METHOD="${1:-task_arithmetic}"
ACTION="${2:-check}"
[[ "$METHOD" == task_arithmetic || "$METHOD" == ties || "$METHOD" == opcm || "$METHOD" == magmax ]] || { echo "[STOP] method must be task_arithmetic, ties, opcm, or magmax" >&2; exit 2; }

TRAIN="$PROJECT_ROOT/data/train_extracted.parquet"
VALIDATION="$PROJECT_ROOT/data/validation_extracted.parquet"
TEST="$PROJECT_ROOT/data/test_extracted.parquet"
FEATURE_COLUMNS="$PROJECT_ROOT/study_assets/data_schema/feature_columns.json"
TASK_FILE="$PROJECT_ROOT/study_assets/task_protocols/head_to_tail_tasks.json"
FOLD_ASSIGNMENTS="$PROJECT_ROOT/study_assets/stratified_3fold_seed42/fold_assignments.parquet"
FOLD_MANIFEST="$PROJECT_ROOT/study_assets/stratified_3fold_seed42/fold_manifest.json"
PREPROCESSING_ROOT="$PROJECT_ROOT/study_assets/preprocessing_ordered_seed0_fold42"
SOURCE_SHARD_CACHE_ROOT="$PROJECT_ROOT/study_assets/source_shard_cache_ordered_seed0_fold42"
MONITOR_ROOT="$PROJECT_ROOT/outputs/${METHOD}_head_to_tail_fixed_head_full_monitor"
LATEST_FILE="$MONITOR_ROOT/latest.txt"

die(){ echo "[STOP] $*" >&2; exit 1; }
require_file(){ [[ -f "$1" ]] || die "Missing file: $1"; }
latest_home(){ require_file "$LATEST_FILE"; local p; p="$(<"$LATEST_FILE")"; [[ -d "$p" ]] || die "Missing run: $p"; printf '%s\n' "$p"; }
pid_running(){ [[ -f "$1" ]] || return 1; local p; p="$(<"$1")"; [[ "$p" =~ ^[0-9]+$ ]] && kill -0 "$p" 2>/dev/null; }

check(){
  cd "$PROJECT_ROOT"
  for p in "$PYTHON" "$TRAIN" "$VALIDATION" "$TEST" "$FEATURE_COLUMNS" "$TASK_FILE" "$FOLD_ASSIGNMENTS" "$FOLD_MANIFEST"; do require_file "$p"; done
  for m in 0 1 2; do require_file "$PREPROCESSING_ROOT/member_${m}/fold_preprocessing.json"; done
  "$PYTHON" -m py_compile src/training/train_cil.py src/training/model_merging_training.py src/training/model_merging_ensemble.py src/methods/model_merging.py
  "$PYTHON" src/training/train_cil.py --help | grep -E "task_arithmetic|ties|opcm|magmax" >/dev/null || die "train_cil does not expose merging methods"
  echo "[OK] $METHOD model-merging plan validated; no training started."
}

member_command(){
  local member="$1" output="$2" prep="$PREPROCESSING_ROOT/member_${1}/fold_preprocessing.json"
  "$PYTHON" "$PROJECT_ROOT/src/training/train_cil.py" \
    --train "$TRAIN" --validation "$VALIDATION" --test "$TEST" \
    --feature-cols "$FEATURE_COLUMNS" --fold-assignments "$FOLD_ASSIGNMENTS" \
    --fold-manifest "$FOLD_MANIFEST" --task-file "$TASK_FILE" \
    --fold-preprocessing "$prep" --preprocessing-policy task0_standard_frozen \
    --source-shard-cache-root "$SOURCE_SHARD_CACHE_ROOT" \
    --member-id "$member" --seed 0 --fold-seed 42 --device cuda \
    --stop-after-task 7 --outdir "$output/member_${member}" \
    --method "$METHOD" --loss-variant cross_entropy \
    --batch-size 64 --effective-batch-size 1024 --epochs 30 --patience 5 \
    --lr 0.001 --weight-decay 0.0001 --dropout 0.2 --activation gelu --norm layernorm \
    --evaluation-batch-size 512
}

worker(){
  local home="$1" output="$2"; mkdir -p "$home" "$output"
  exec > >(tee -a "$home/main.log") 2>&1
  set +e
  if [[ "$METHOD" == opcm ]]; then
    echo "[RECIPE] OPCM paper-faithful online train-current-then-merge + shared-initialization experts + TDDI + fixed 178-class head + head-to-tail"
  elif [[ "$METHOD" == magmax ]]; then
    echo "[RECIPE] MagMax paper-faithful online sequential fine-tuning + immediate maximum-magnitude merge (lambda=0.5) + TDDI + fixed 178-class head + head-to-tail"
  else
    echo "[RECIPE] $METHOD + independent experts + TDDI + fixed 178-class head + head-to-tail"
  fi
  for member in 0 1 2; do
    if [[ "$METHOD" == opcm ]]; then
      echo "[RUN] member=$member; train task t from shared initialization, merge immediately, then expose task t+1"
    elif [[ "$METHOD" == magmax ]]; then
      echo "[RUN] member=$member; train task t, merge immediately, then continue to task t+1"
    else
      echo "[RUN] member=$member; each expert resets from task-0 base"
    fi
    member_command "$member" "$output"; code=$?
    if [[ $code -ne 0 ]]; then
      printf '%s\n' "$code" > "$home/exit_code.txt"
      printf '%s\n' "$code" > "$output/runner_exit_code.txt"
      echo "[FAILED] member=$member exit_code=$code"
      exit "$code"
    fi
  done
  PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" "$PROJECT_ROOT/src/training/model_merging_ensemble.py" --root "$output" --method "$METHOD"; code=$?
  if [[ $code -eq 0 ]]; then sha256sum "$output/${output##*/}_metrics_review.zip" > "$output/${output##*/}_metrics_review.zip.sha256"; fi
  printf '%s\n' "$code" > "$home/exit_code.txt"
  printf '%s\n' "$code" > "$output/runner_exit_code.txt"
  [[ $code -eq 0 ]] && echo "[COMPLETED] $METHOD independent-expert full run and review package ready." || echo "[FAILED] ensemble exit_code=$code"
  exit "$code"
}

start(){
  cd "$PROJECT_ROOT"; check; mkdir -p "$MONITOR_ROOT"
  local id home output; id="$(date -u +%Y%m%dT%H%M%SZ)"
  home="$MONITOR_ROOT/run_${id}"; output="$PROJECT_ROOT/outputs/${METHOD}_head_to_tail_fixed_head_full_seed0_${id}"
  mkdir -p "$home" "$output"; printf '%s\n' "$output" > "$home/output_root.txt"; printf '%s\n' "$home" > "$LATEST_FILE"
  nohup bash "$0" "$METHOD" _worker "$home" "$output" > "$home/nohup.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$home/job.pid"; echo "[STARTED] PID=$(<$home/job.pid)"; echo "[LOG] $home/main.log"
}

status(){
  local home; home="$(latest_home)"; local output; output="$(<"$home/output_root.txt")"
  if pid_running "$home/job.pid"; then echo "[RUNNING] PID=$(<$home/job.pid)"; elif [[ -f "$home/exit_code.txt" ]]; then echo "[FINISHED] exit_code=$(<$home/exit_code.txt)"; else echo "[NOT RUNNING]"; fi
  echo "Output: $output"; [[ -f "$output/ensemble_summary.json" ]] && echo "Offline ensemble: completed" || echo "Offline ensemble: pending"
  [[ -f "$output/${output##*/}_metrics_review.zip" ]] && echo "Review ZIP: ready" || echo "Review ZIP: pending"
}

if [[ "$ACTION" == _worker ]]; then worker "$3" "$4"; fi
case "$ACTION" in
  check) check;;
  start) start;;
  status) status;;
  follow) tail -n 100 -F "$(latest_home)/main.log";;
  *) echo "Usage: bash scripts/run_model_merging_full.sh {task_arithmetic|ties|opcm|magmax} {check|start|status|follow}"; exit 2;;
esac
