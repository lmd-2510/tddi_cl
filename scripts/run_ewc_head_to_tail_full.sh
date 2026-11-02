#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$PROJECT_ROOT/scripts/run_ewc_head_to_tail_full.sh"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

CONFIG="$PROJECT_ROOT/configs/ordered_baseline.json"
TRAIN="$PROJECT_ROOT/data/train_extracted.parquet"
VALIDATION="$PROJECT_ROOT/data/validation_extracted.parquet"
TEST="$PROJECT_ROOT/data/test_extracted.parquet"
FEATURE_COLUMNS="$PROJECT_ROOT/study_assets/data_schema/feature_columns.json"
TASK_FILE="$PROJECT_ROOT/study_assets/task_protocols/head_to_tail_tasks.json"
FOLD_ASSIGNMENTS="$PROJECT_ROOT/study_assets/stratified_3fold_seed42/fold_assignments.parquet"
FOLD_MANIFEST="$PROJECT_ROOT/study_assets/stratified_3fold_seed42/fold_manifest.json"
PREPROCESSING_ROOT="$PROJECT_ROOT/study_assets/preprocessing_ordered_seed0_fold42"
SOURCE_SHARD_CACHE_ROOT="$PROJECT_ROOT/study_assets/source_shard_cache_ordered_seed0_fold42"
COMMON_TASK_CACHE_ROOT="$PROJECT_ROOT/study_assets/common_task_cache_ordered_seed0_fold42"

MONITOR_ROOT="$PROJECT_ROOT/outputs/ewc_head_to_tail_full_monitor"
LATEST_FILE="$MONITOR_ROOT/latest.txt"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_ewc_head_to_tail_full.sh check
  bash scripts/run_ewc_head_to_tail_full.sh start
  bash scripts/run_ewc_head_to_tail_full.sh status
  bash scripts/run_ewc_head_to_tail_full.sh follow
  bash scripts/run_ewc_head_to_tail_full.sh resume
  bash scripts/run_ewc_head_to_tail_full.sh package

Commands:
  check    Validate CUDA, data, folds, preprocessing, cache and the dry-run plan.
  start    Start a new detached full run with nohup.
  status   Show the latest job and per-member progress.
  follow   Follow the latest main log (Ctrl-C only stops following the log).
  resume   Resume the latest interrupted run from valid task checkpoints.
  package  Rebuild report, visualization and lightweight review ZIP only.

Recipe:
  protocol=head_to_tail, backbone=TDDI, method=EWC, fixed 178-class head,
  members=0,1,2 sequentially, tasks=0..7, then offline ensemble.
EOF
}

die() {
  echo "[STOP] $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "Missing file: $1"
}

latest_run_home() {
  require_file "$LATEST_FILE"
  local run_home
  run_home="$(<"$LATEST_FILE")"
  [[ -d "$run_home" ]] || die "Latest monitor directory does not exist: $run_home"
  printf '%s\n' "$run_home"
}

pid_is_running() {
  local pid_file="$1"
  [[ -f "$pid_file" ]] || return 1
  local pid
  pid="$(<"$pid_file")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

full_command() {
  local output_root="$1"
  local execute_flag="${2:-}"
  local command=(
    "$PYTHON" "$PROJECT_ROOT/src/training/fold_ensemble3_full.py"
    --config "$CONFIG"
    --method ewc
    --train "$TRAIN"
    --validation "$VALIDATION"
    --test "$TEST"
    --feature-cols "$FEATURE_COLUMNS"
    --fold-assignments "$FOLD_ASSIGNMENTS"
    --fold-manifest "$FOLD_MANIFEST"
    --task-file "$TASK_FILE"
    --preprocessing-root "$PREPROCESSING_ROOT"
    --output-root "$output_root"
    --device cuda
    --python "$PYTHON"
  )
  if [[ "$execute_flag" == "execute" ]]; then
    command+=(--execute)
  fi
  printf '%q ' "${command[@]}"
  printf '\n'
}

check_inputs() {
  local required=(
    "$PYTHON"
    "$CONFIG"
    "$TRAIN"
    "$VALIDATION"
    "$TEST"
    "$FEATURE_COLUMNS"
    "$TASK_FILE"
    "$FOLD_ASSIGNMENTS"
    "$FOLD_MANIFEST"
    "$SOURCE_SHARD_CACHE_ROOT/manifest.json"
  )
  local member
  for member in 0 1 2; do
    required+=("$PREPROCESSING_ROOT/member_${member}/fold_preprocessing.json")
  done
  local path
  for path in "${required[@]}"; do
    require_file "$path"
  done

  "$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("[STOP] PyTorch cannot access CUDA.")
name = torch.cuda.get_device_name(0)
free, total = torch.cuda.mem_get_info(0)
print(f"[OK] CUDA: {name}; free={free / 1024**3:.2f} GiB; total={total / 1024**3:.2f} GiB")
PY
}

check_all() {
  cd "$PROJECT_ROOT"
  check_inputs
  export CIL_SOURCE_SHARD_CACHE_ROOT="$SOURCE_SHARD_CACHE_ROOT"
  export CIL_COMMON_TASK_CACHE_ROOT="$COMMON_TASK_CACHE_ROOT"

  local dry_root="$PROJECT_ROOT/outputs/_ewc_head_to_tail_dry_run_only"
  echo "[CHECK] Validating the immutable full-run plan; no model will be trained."
  local command
  command="$(full_command "$dry_root")"
  eval "$command"
  echo "[OK] EWC full-run preflight passed."
  echo "[OK] Common method-independent cache: $COMMON_TASK_CACHE_ROOT"
}

postprocess() {
  local output_root="$1"
  echo "[POST] Building final metric tables, BWT and forgetting summaries."
  "$PYTHON" "$PROJECT_ROOT/src/eval/report.py" \
    --full-root "$output_root" \
    --task-file "$TASK_FILE" \
    --overwrite

  echo "[POST] Building final-task ensemble figures and the task 0-7 forgetting heatmap."
  "$PYTHON" "$PROJECT_ROOT/scripts/visualize_cil_run.py" \
    --run-root "$output_root" \
    --outdir "$output_root/visualizations_ensemble" \
    --task-id 7 \
    --member-id 0 \
    --member-ids 0 1 2 \
    --ensemble \
    --split test \
    --task-file "$TASK_FILE" \
    --train "$TRAIN" \
    --validation "$VALIDATION" \
    --test "$TEST"

  local archive="$output_root/${output_root##*/}_metrics_review.zip"
  echo "[POST] Packaging lightweight results only (no NPZ, checkpoints or Parquet)."
  "$PYTHON" -m src.eval.package \
    --run-root "$output_root" \
    --archive "$archive" \
    --overwrite
}

worker() {
  local run_home="$1"
  local output_root="$2"
  mkdir -p "$run_home" "$output_root" "$COMMON_TASK_CACHE_ROOT"
  exec > >(tee -a "$run_home/main.log") 2>&1

  set +e
  local attempt=1
  if [[ -f "$run_home/runner_attempts.txt" ]]; then
    attempt=$(( $(<"$run_home/runner_attempts.txt") + 1 ))
  fi
  printf '%s\n' "$attempt" > "$run_home/runner_attempts.txt"
  echo "[RUNNER_ATTEMPT] $attempt"
  echo "[OUTPUT] $output_root"
  echo "[RECIPE] EWC + TDDI + fixed 178-class head + head-to-tail + 3 sequential members"

  export CIL_SOURCE_SHARD_CACHE_ROOT="$SOURCE_SHARD_CACHE_ROOT"
  export CIL_COMMON_TASK_CACHE_ROOT="$COMMON_TASK_CACHE_ROOT"

  local command
  command="$(full_command "$output_root" execute)"
  eval "$command"
  local exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    postprocess "$output_root"
    exit_code=$?
  fi

  printf '%s\n' "$exit_code" > "$run_home/exit_code.txt"
  printf '%s\n' "$exit_code" > "$output_root/runner_exit_code.txt"
  if [[ $exit_code -eq 0 ]]; then
    echo "[COMPLETED] Full EWC run, offline ensemble and review package are ready."
    echo "[REVIEW] $output_root/${output_root##*/}_metrics_review.zip"
  else
    echo "[FAILED] exit_code=$exit_code; inspect $run_home/main.log"
  fi
  exit "$exit_code"
}

start_new() {
  cd "$PROJECT_ROOT"
  check_all
  mkdir -p "$MONITOR_ROOT"
  if [[ -f "$LATEST_FILE" ]]; then
    local previous
    previous="$(<"$LATEST_FILE")"
    if [[ -d "$previous" ]] && pid_is_running "$previous/job.pid"; then
      die "A previous launcher is still running (PID $(<"$previous/job.pid"))."
    fi
  fi

  local run_id run_home output_root
  run_id="$(date -u +%Y%m%dT%H%M%SZ)"
  run_home="$MONITOR_ROOT/run_${run_id}"
  output_root="$PROJECT_ROOT/outputs/ewc_head_to_tail_fixed_head_full_seed0_${run_id}"
  mkdir -p "$run_home" "$output_root"
  printf '%s\n' "$run_id" > "$run_home/run_id.txt"
  printf '%s\n' "$output_root" > "$run_home/output_root.txt"
  printf '%s\n' "$run_home" > "$MONITOR_ROOT/.latest.tmp"
  mv "$MONITOR_ROOT/.latest.tmp" "$LATEST_FILE"

  nohup bash "$SCRIPT_PATH" _worker "$run_home" "$output_root" \
    > "$run_home/nohup.log" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "$pid" > "$run_home/job.pid"
  echo "[STARTED] PID=$pid"
  echo "[OUTPUT] $output_root"
  echo "[LOG] $run_home/main.log"
  echo "You may close SSH after 'bash scripts/run_ewc_head_to_tail_full.sh status' reports RUNNING."
}

show_status() {
  cd "$PROJECT_ROOT"
  local run_home output_root
  run_home="$(latest_run_home)"
  require_file "$run_home/output_root.txt"
  output_root="$(<"$run_home/output_root.txt")"
  local pid="unknown"
  [[ -f "$run_home/job.pid" ]] && pid="$(<"$run_home/job.pid")"
  if pid_is_running "$run_home/job.pid"; then
    echo "[RUNNING] PID=$pid"
  elif [[ -f "$run_home/exit_code.txt" ]]; then
    local exit_code
    exit_code="$(<"$run_home/exit_code.txt")"
    [[ "$exit_code" == "0" ]] && echo "[COMPLETED] exit_code=0" || echo "[FAILED] exit_code=$exit_code"
  else
    echo "[NOT RUNNING] PID=$pid; no final exit code was recorded."
  fi
  echo "Output: $output_root"

  local member latest_checkpoint
  for member in 0 1 2; do
    latest_checkpoint=""
    if [[ -d "$output_root/member_${member}/checkpoints" ]]; then
      latest_checkpoint="$(find "$output_root/member_${member}/checkpoints" -maxdepth 1 -type f -name 'task_*.pt' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1)"
    fi
    if [[ -n "$latest_checkpoint" ]]; then
      echo "M${member}: latest checkpoint=$latest_checkpoint"
    else
      echo "M${member}: not started"
    fi
  done
  [[ -f "$output_root/full_manifest.json" ]] && echo "Offline ensemble: completed" || echo "Offline ensemble: pending"
  [[ -d "$output_root/final_results" ]] && echo "Final report: completed" || echo "Final report: pending"
  local archive="$output_root/${output_root##*/}_metrics_review.zip"
  [[ -f "$archive" ]] && echo "Review ZIP: $archive" || echo "Review ZIP: pending"
}

follow_log() {
  local run_home
  run_home="$(latest_run_home)"
  touch "$run_home/main.log"
  tail -n 100 -F "$run_home/main.log"
}

resume_latest() {
  cd "$PROJECT_ROOT"
  check_inputs
  local run_home output_root
  run_home="$(latest_run_home)"
  require_file "$run_home/output_root.txt"
  output_root="$(<"$run_home/output_root.txt")"
  if pid_is_running "$run_home/job.pid"; then
    die "The latest launcher is already running (PID $(<"$run_home/job.pid"))."
  fi
  rm -f "$run_home/exit_code.txt" "$output_root/runner_exit_code.txt"
  nohup bash "$SCRIPT_PATH" _worker "$run_home" "$output_root" \
    > "$run_home/nohup.log" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "$pid" > "$run_home/job.pid"
  echo "[RESUMED] PID=$pid"
  echo "[OUTPUT] $output_root"
  echo "[LOG] $run_home/main.log"
}

package_latest() {
  cd "$PROJECT_ROOT"
  check_inputs
  local run_home output_root
  run_home="$(latest_run_home)"
  require_file "$run_home/output_root.txt"
  output_root="$(<"$run_home/output_root.txt")"
  [[ -f "$output_root/full_manifest.json" ]] || die "Offline ensemble is incomplete: $output_root"
  postprocess "$output_root"
}

command="${1:-}"
case "$command" in
  check) check_all ;;
  start) start_new ;;
  status) show_status ;;
  follow) follow_log ;;
  resume) resume_latest ;;
  package) package_latest ;;
  _worker)
    [[ $# -eq 3 ]] || die "Invalid internal worker arguments."
    worker "$2" "$3"
    ;;
  -h|--help|help|"") usage ;;
  *) usage; die "Unknown command: $command" ;;
esac
