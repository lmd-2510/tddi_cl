#!/usr/bin/env bash
set -euo pipefail

# Paper-style Experience Replay (reservoir memory, 1:1 replay, CE) on the
# existing frozen-fold/head-to-tail workflow.  This launcher is independent of
# the EWC/A-GEM launcher so its method contract cannot accidentally change.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$PROJECT_ROOT/scripts/run_er_head_to_tail_full.sh"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
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
MONITOR_ROOT="$PROJECT_ROOT/outputs/er_head_to_tail_full_monitor"
LATEST_FILE="$MONITOR_ROOT/latest.txt"

die(){ echo "[STOP] $*" >&2; exit 1; }
require_file(){ [[ -f "$1" ]] || die "Missing file: $1"; }
latest_run_home(){ require_file "$LATEST_FILE"; local p; p="$(<"$LATEST_FILE")"; [[ -d "$p" ]] || die "Missing run: $p"; printf '%s\n' "$p"; }
pid_running(){ [[ -f "$1" ]] || return 1; local p; p="$(<"$1")"; [[ "$p" =~ ^[0-9]+$ ]] && kill -0 "$p" 2>/dev/null; }

full_command(){
  local output="$1" flag="${2:-}"
  local c=("$PYTHON" "$PROJECT_ROOT/src/training/fold_ensemble3_full.py"
    --config "$CONFIG" --method er --train "$TRAIN" --validation "$VALIDATION" --test "$TEST"
    --feature-cols "$FEATURE_COLUMNS" --fold-assignments "$FOLD_ASSIGNMENTS"
    --fold-manifest "$FOLD_MANIFEST" --task-file "$TASK_FILE"
    --preprocessing-root "$PREPROCESSING_ROOT" --output-root "$output"
    --device cuda --python "$PYTHON")
  [[ "$flag" == execute ]] && c+=(--execute)
  printf '%q ' "${c[@]}"; printf '\n'
}

check_inputs(){
  local paths=("$PYTHON" "$CONFIG" "$TRAIN" "$VALIDATION" "$TEST" "$FEATURE_COLUMNS" "$TASK_FILE"
    "$FOLD_ASSIGNMENTS" "$FOLD_MANIFEST" "$SOURCE_SHARD_CACHE_ROOT/manifest.json")
  local m p; for m in 0 1 2; do paths+=("$PREPROCESSING_ROOT/member_${m}/fold_preprocessing.json"); done
  for p in "${paths[@]}"; do require_file "$p"; done
  "$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available(): raise SystemExit("[STOP] CUDA is unavailable")
free,total=torch.cuda.mem_get_info(0)
print(f"[OK] CUDA: {torch.cuda.get_device_name(0)}; free={free/1024**3:.2f} GiB; total={total/1024**3:.2f} GiB")
PY
}

check_all(){
  cd "$PROJECT_ROOT"; check_inputs
  export CIL_SOURCE_SHARD_CACHE_ROOT="$SOURCE_SHARD_CACHE_ROOT" CIL_COMMON_TASK_CACHE_ROOT="$COMMON_TASK_CACHE_ROOT"
  local dry="$PROJECT_ROOT/outputs/_er_head_to_tail_dry_run_only" cmd
  echo "[CHECK] Validating ER frozen-fold plan; no model will be trained."
  cmd="$(full_command "$dry")"; eval "$cmd"
  echo "[OK] ER plan passed: reservoir_uniform_v1 + uniform replay with equal current/replay batch + CE."
}

postprocess(){
  local output="$1"
  "$PYTHON" "$PROJECT_ROOT/src/eval/report.py" --full-root "$output" --task-file "$TASK_FILE" --overwrite
  "$PYTHON" "$PROJECT_ROOT/scripts/visualize_cil_run.py" --run-root "$output" \
    --outdir "$output/visualizations_ensemble" --task-id 7 --member-id 0 --member-ids 0 1 2 \
    --ensemble --split test --task-file "$TASK_FILE" --train "$TRAIN" --validation "$VALIDATION" --test "$TEST"
  local archive="$output/${output##*/}_metrics_review.zip"
  "$PYTHON" -m src.eval.package --run-root "$output" --archive "$archive" --overwrite
  echo "[REVIEW] $archive"
}

worker(){
  local home="$1" output="$2"; mkdir -p "$home" "$output" "$COMMON_TASK_CACHE_ROOT"
  exec > >(tee -a "$home/main.log") 2>&1
  export CIL_SOURCE_SHARD_CACHE_ROOT="$SOURCE_SHARD_CACHE_ROOT" CIL_COMMON_TASK_CACHE_ROOT="$COMMON_TASK_CACHE_ROOT"
  set +e; echo "[RECIPE] ER + TDDI + fixed 178-class head + head-to-tail + members 0,1,2"
  local cmd="$(full_command "$output" execute)"; eval "$cmd"; local code=$?
  if [[ $code -eq 0 ]]; then postprocess "$output"; code=$?; fi
  printf '%s\n' "$code" > "$home/exit_code.txt"; printf '%s\n' "$code" > "$output/runner_exit_code.txt"
  [[ $code -eq 0 ]] && echo "[COMPLETED] ER full run, offline ensemble and review package ready." || echo "[FAILED] exit_code=$code; inspect $home/main.log"
  exit "$code"
}

start_new(){
  cd "$PROJECT_ROOT"; check_all; mkdir -p "$MONITOR_ROOT"
  if [[ -f "$LATEST_FILE" ]]; then local old; old="$(<"$LATEST_FILE")"; pid_running "$old/job.pid" && die "Previous ER launcher is still running (PID $(<"$old/job.pid"))."; fi
  local id home output; id="$(date -u +%Y%m%dT%H%M%SZ)"; home="$MONITOR_ROOT/run_${id}"; output="$PROJECT_ROOT/outputs/er_head_to_tail_fixed_head_full_seed0_${id}"
  mkdir -p "$home" "$output"; printf '%s\n' "$output" > "$home/output_root.txt"; printf '%s\n' "$home" > "$LATEST_FILE"
  nohup bash "$SCRIPT_PATH" _worker "$home" "$output" > "$home/nohup.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$home/job.pid"; echo "[STARTED] PID=$(<$home/job.pid)"; echo "[LOG] $home/main.log"
}

status(){
  local home output; home="$(latest_run_home)"; output="$(<"$home/output_root.txt")"
  if pid_running "$home/job.pid"; then echo "[RUNNING] PID=$(<$home/job.pid)"; elif [[ -f "$home/exit_code.txt" ]]; then echo "[FINISHED] exit_code=$(<$home/exit_code.txt)"; else echo "[NOT RUNNING]"; fi
  echo "Output: $output"; local m ck; for m in 0 1 2; do ck="$(find "$output/member_${m}/checkpoints" -maxdepth 1 -name 'task_*.pt' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1)"; echo "M$m: ${ck:-not started}"; done
  [[ -f "$output/full_manifest.json" ]] && echo "Offline ensemble: completed" || echo "Offline ensemble: pending"
  [[ -f "$output/${output##*/}_metrics_review.zip" ]] && echo "Review ZIP: ready" || echo "Review ZIP: pending"
}

resume(){
  local home output; home="$(latest_run_home)"; output="$(<"$home/output_root.txt")"; pid_running "$home/job.pid" && die "ER run is still running"; rm -f "$home/exit_code.txt" "$output/runner_exit_code.txt"
  nohup bash "$SCRIPT_PATH" _worker "$home" "$output" > "$home/nohup.log" 2>&1 < /dev/null & printf '%s\n' "$!" > "$home/job.pid"; echo "[RESUMED] PID=$(<$home/job.pid)"
}

package_latest(){
  cd "$PROJECT_ROOT"; check_inputs
  local home output; home="$(latest_run_home)"; output="$(<"$home/output_root.txt")"
  [[ -f "$output/full_manifest.json" ]] || die "Offline ensemble is incomplete: $output"
  postprocess "$output"
}

case "${1:-}" in
  check) check_all;; start) start_new;; status) status;; follow) tail -n 100 -F "$(latest_run_home)/main.log";; resume) check_inputs; resume;; package) package_latest;;
  _worker) [[ $# -eq 3 ]] || die "invalid worker args"; worker "$2" "$3";;
  *) echo "Usage: bash scripts/run_er_head_to_tail_full.sh {check|start|status|follow|resume|package}"; exit 2;;
esac
