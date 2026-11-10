#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$PROJECT_ROOT/scripts/run_ddinet_model_merging_full.sh"
PYTHON="${CIL_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
METHOD="${1:-}"
ACTION="${2:-check}"
case "$METHOD" in
  task_arithmetic|ties) ;;
  *) echo "Usage: $0 {task_arithmetic|ties} {check|start|status|follow|package}"; exit 2 ;;
esac

TRAIN="$PROJECT_ROOT/data/train_extracted.parquet"
VALIDATION="$PROJECT_ROOT/data/validation_extracted.parquet"
TEST="$PROJECT_ROOT/data/test_extracted.parquet"
FEATURE_COLUMNS="$PROJECT_ROOT/study_assets/data_schema/feature_columns.json"
TASK_FILE="$PROJECT_ROOT/study_assets/task_protocols/head_to_tail_tasks.json"
FOLD_ROOT="$PROJECT_ROOT/study_assets/stratified_3fold_seed42"
PREPROCESSING_ROOT="$PROJECT_ROOT/study_assets/preprocessing_ddinet_seed0_fold42"
SOURCE_CACHE="$PROJECT_ROOT/study_assets/source_shard_cache_ordered_seed0_fold42"
FINGERPRINT_CACHE="$PROJECT_ROOT/study_assets/ddinet_fingerprint_cache_morgan1024"
MONITOR_ROOT="$PROJECT_ROOT/outputs/${METHOD}_ddinet_head_to_tail_full_monitor"

die() { echo "[STOP] $*" >&2; exit 1; }
latest_home() {
  [[ -f "$MONITOR_ROOT/latest.txt" ]] || die "No previous run."
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

check() {
  cd "$PROJECT_ROOT"
  # This action reuses the existing DDINet preparation entry point.
  CIL_PYTHON="$PYTHON" bash scripts/run_ewc_ddinet_head_to_tail_full.sh prepare
  "$PYTHON" - "$PROJECT_ROOT" "$METHOD" <<'PY'
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from src.data.ddinet_fingerprints import DDINetFingerprintConfig, load_ddinet_fingerprint_cache
from src.data.ddi_dataset import load_feature_columns, prepare_development_fold_context
from src.data.fold_preprocessing import load_fold_preprocessing
from src.data.source_shard_cache import validate_source_shard_cache
from src.training.fold_training import validate_fold_task_spec
from src.training import train_cil

root = Path(sys.argv[1])
sources = {name: root / "data" / f"{name}_extracted.parquet" for name in ("train", "validation", "test")}
folds = root / "study_assets/stratified_3fold_seed42"
task_file = root / "study_assets/task_protocols/head_to_tail_tasks.json"
columns = load_feature_columns(root / "study_assets/data_schema/feature_columns.json")
context = prepare_development_fold_context(folds / "fold_assignments.parquet", folds / "fold_manifest.json", source_paths=sources)
spec = train_cil.load_task_spec(task_file)
if spec["protocol"] != "head_to_tail":
    raise SystemExit("Expected head_to_tail protocol")
validate_fold_task_spec(spec, context)
for member in (0, 1, 2):
    load_fold_preprocessing(
        root / f"study_assets/preprocessing_ddinet_seed0_fold42/member_{member}/fold_preprocessing.json",
        context=context, task_file=task_file, feature_columns=columns,
        member_id=member, validation_fold=member, policy="raw_identity", experiment_seed=0,
    )
validate_source_shard_cache(
    root / "study_assets/source_shard_cache_ordered_seed0_fold42", task_file=task_file,
    feature_columns=columns, fold_manifest_sha256=context.manifest_sha256,
    assignment_sha256=context.manifest["assignment_sha256"],
    source_paths={name: sources[name] for name in ("train", "validation")},
)
cache = load_ddinet_fingerprint_cache(root / "study_assets/ddinet_fingerprint_cache_morgan1024",
                                      expected_config=DDINetFingerprintConfig())
assert cache.input_dim == 2048
model = train_cil.model_manifest(SimpleNamespace(variant="ddinet_multiclass_cil", method=sys.argv[2], dropout=.3, activation="relu"))
assert model["hidden_dims"][-1] == 128
assert model["singleton_batchnorm_policy"] == "running_stats_for_singleton_v1"
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
print(f"[OK] CUDA {torch.cuda.get_device_name(0)}")
print("[OK] P1, folds, raw-identity artifacts, source shards, Morgan cache and DDINet policy validated.", flush=True)
PY
}

member_command() {
  local member="$1" output="$2"
  "$PYTHON" "$PROJECT_ROOT/src/training/train_cil.py" \
    --train "$TRAIN" --validation "$VALIDATION" --test "$TEST" \
    --feature-cols "$FEATURE_COLUMNS" --task-file "$TASK_FILE" \
    --fold-assignments "$FOLD_ROOT/fold_assignments.parquet" --fold-manifest "$FOLD_ROOT/fold_manifest.json" \
    --fold-preprocessing "$PREPROCESSING_ROOT/member_${member}/fold_preprocessing.json" \
    --preprocessing-policy raw_identity --source-shard-cache-root "$SOURCE_CACHE" \
    --ddinet-fingerprint-cache "$FINGERPRINT_CACHE" --variant ddinet_multiclass_cil \
    --member-id "$member" --seed 0 --fold-seed 42 --device cuda --stop-after-task 7 \
    --outdir "$output/member_${member}" --method "$METHOD" --loss-variant cross_entropy \
    --batch-size 64 --effective-batch-size 1024 --evaluation-batch-size 512 \
    --epochs 30 --patience 5 --lr 0.001 --weight-decay 0.0001 \
    --dropout 0.3 --activation relu --norm none
}

package_output() {
  local output="$1"
  "$PYTHON" "$PROJECT_ROOT/src/training/model_merging_ensemble.py" --root "$output" --method "$METHOD" || return $?
  sha256sum "$output/${output##*/}_metrics_review.zip" > "$output/${output##*/}_metrics_review.zip.sha256"
}

worker() {
  local run_home="$1" output="$2"
  cd "$PROJECT_ROOT"
  exec > >(tee -a "$run_home/main.log") 2>&1
  set +e
  echo "[RECIPE] $METHOD + DDINet 128-d + fixed 178 head + head-to-tail + members 0,1,2"
  echo "[MODEL] Morgan 2048-d, dropout 0.3, ReLU, BatchNorm singleton running-stat policy"
  echo "[MERGE] Independent task-0-base experts, owned head rows, pooled BatchNorm moments"
  local code=0
  for member in 0 1 2; do
    echo "[RUN] member=$member"
    member_command "$member" "$output"
    code=$?
    [[ $code -eq 0 ]] || break
  done
  if [[ $code -eq 0 ]]; then
    package_output "$output"
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

start() {
  check
  mkdir -p "$MONITOR_ROOT"
  if [[ -f "$MONITOR_ROOT/latest.txt" ]]; then
    local previous
    previous="$(latest_home)"
    if running "$previous"; then die "A launcher for this method is already running."; fi
  fi
  local run_id run_home output
  run_id="$(date -u +%Y%m%dT%H%M%SZ)"
  run_home="$MONITOR_ROOT/run_${run_id}"
  output="$PROJECT_ROOT/outputs/${METHOD}_ddinet_head_to_tail_fixed_head_full_seed0_${run_id}"
  mkdir -p "$run_home" "$output"
  printf '%s\n' "$run_home" > "$MONITOR_ROOT/latest.txt"
  printf '%s\n' "$output" > "$run_home/output_root.txt"
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
    if [[ -f "$output/member_${member}/run_summary.json" ]]; then
      echo "M${member}: completed"
    else
      echo "M${member}: pending or in progress"
    fi
  done
}

case "$ACTION" in
  check) check ;;
  start) start ;;
  status) status ;;
  follow) tail -n 100 -F "$(latest_home)/main.log" ;;
  package)
    run_home="$(latest_home)"
    if running "$run_home"; then die "Wait for training to finish before packaging."; fi
    package_output "$(<"$run_home/output_root.txt")"
    ;;
  _worker) worker "$3" "$4" ;;
  *) die "Unknown action: $ACTION" ;;
esac
