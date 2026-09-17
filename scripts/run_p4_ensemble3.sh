#!/usr/bin/env bash
set -uo pipefail

# P4 controller: member 0 -> 1 -> 2 -> offline ensemble/UE/threshold.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${P4_PYTHON:-python}"
GPU_ID="${P4_GPU_ID:-0}"
EXPECTED_TASK_SHA256="9d22af8618fe03c40b92e2aabbab8b68a89de83bc0de4f51e24932951cf298f4"
TRAIN="$REPO_ROOT/train_extracted.parquet"
VALIDATION="$REPO_ROOT/validation_extracted.parquet"
TEST="$REPO_ROOT/test_extracted.parquet"
FEATURES="$REPO_ROOT/study_assets/data_schema/feature_columns.json"
TASK_FILE="$REPO_ROOT/study_assets/task_protocols/constrained_mass_balanced_seed0_tasks.json"
PREP_ROOT="$REPO_ROOT/study_assets/preprocessing_p4_seed0_fold42"
FULL_CONFIG="${P4_FULL_CONFIG:-$REPO_ROOT/configs/full_tddi_p4_fold_ensemble3_seed0.json}"
THRESHOLD_CONFIG="${P4_THRESHOLD_CONFIG:-$REPO_ROOT/configs/eval_tddi_p4_ensemble_entropy_balanced_accuracy_threshold.json}"
FULL_ROOT="${P4_FULL_ROOT:-$REPO_ROOT/outputs/stratified_ensemble3/full_p4_seed0_8tasks}"
MONITOR_ROOT="${P4_MONITOR_ROOT:-$REPO_ROOT/outputs/stratified_ensemble3_monitor/full_p4_seed0_8tasks}"
CONTROLLER_SCRIPT="${P4_CONTROLLER_SCRIPT:-scripts/run_p4_ensemble3.sh}"

die() { echo "[STOP] $*" >&2; exit 1; }

resolve_fold_root() {
  if [[ -n "${P4_FOLD_ROOT:-}" ]]; then
    printf '%s\n' "$P4_FOLD_ROOT"
    return
  fi
  local preferred="$REPO_ROOT/outputs/fold_preparation_seed42_20260912_161002/folds"
  if [[ -s "$preferred/fold_assignments.parquet" && -s "$preferred/fold_manifest.json" ]]; then
    printf '%s\n' "$preferred"
    return
  fi
  local candidates=() candidate
  shopt -s nullglob
  for candidate in "$REPO_ROOT"/outputs/fold_preparation_seed42_*/folds; do
    if [[ -s "$candidate/fold_assignments.parquet" && -s "$candidate/fold_manifest.json" ]]; then
      candidates+=("$candidate")
    fi
  done
  shopt -u nullglob
  (( ${#candidates[@]} == 1 )) \
    || die "Không xác định được duy nhất fold root. Hãy export P4_FOLD_ROOT=/duong/dan/folds"
  printf '%s\n' "${candidates[0]}"
}

FOLD_ROOT="$(resolve_fold_root)"
ASSIGNMENTS="$FOLD_ROOT/fold_assignments.parquet"
FOLD_MANIFEST="$FOLD_ROOT/fold_manifest.json"
COMMON_P4=(
  --train "$TRAIN" --validation "$VALIDATION" --test "$TEST"
  --feature-cols "$FEATURES"
  --fold-assignments "$ASSIGNMENTS" --fold-manifest "$FOLD_MANIFEST"
  --task-file "$TASK_FILE" --preprocessing-root "$PREP_ROOT"
  --output-root "$FULL_ROOT" --threshold-config "$THRESHOLD_CONFIG"
  --device cuda
)

check_inputs() {
  local file
  for file in "$TRAIN" "$VALIDATION" "$TEST" "$FEATURES" \
    "$ASSIGNMENTS" "$FOLD_MANIFEST" "$TASK_FILE" "$FULL_CONFIG" "$THRESHOLD_CONFIG"; do
    [[ -s "$file" ]] || die "Thiếu input: $file"
  done
  local actual_hash
  actual_hash="$(sha256sum "$TASK_FILE" | awk '{print $1}')"
  [[ "$actual_hash" == "$EXPECTED_TASK_SHA256" ]] || die "Sai task-file SHA256: $actual_hash"
  "$PYTHON_BIN" - "$TASK_FILE" <<'PY'
import json, sys
from pathlib import Path
spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
layout = [len(task["classes"]) for task in spec["tasks"]]
classes = [raw for task in spec["tasks"] for raw in task["classes"]]
if layout != [38, 20, 20, 20, 20, 20, 20, 20]:
    raise SystemExit(f"Unexpected P4 layout: {layout}")
if len(classes) != 178 or len(set(classes)) != 178:
    raise SystemExit("P4 must contain exactly 178 unique classes.")
print("[OK] P4 inputs, hash, layout and 178 classes")
PY
  echo "[OK] fold root: $FOLD_ROOT"
}

check_preprocessing() {
  local member_id file
  for member_id in 0 1 2; do
    file="$PREP_ROOT/member_${member_id}/B/fold_preprocessing.json"
    [[ -s "$file" ]] || die "Thiếu preprocessing member $member_id: $file"
  done
  echo "[OK] preprocessing member 0, 1, 2"
}

prepare_preprocessing() {
  check_inputs
  local member_id prep_out prep_file
  for member_id in 0 1 2; do
    prep_out="$PREP_ROOT/member_${member_id}/B"
    prep_file="$prep_out/fold_preprocessing.json"
    if [[ -s "$prep_file" ]]; then
      echo "[SKIP] preprocessing member $member_id đã tồn tại"
      continue
    fi
    [[ ! -e "$prep_out" ]] || die "Namespace preprocessing dở dang: $prep_out"
    "$PYTHON_BIN" scripts/prepare_fold_preprocessing.py \
      --assignments "$ASSIGNMENTS" --manifest "$FOLD_MANIFEST" \
      --train "$TRAIN" --validation "$VALIDATION" --test "$TEST" \
      --task-file "$TASK_FILE" --feature-cols "$FEATURES" \
      --member-id "$member_id" --validation-fold "$member_id" \
      --experiment-seed 0 --fold-seed 42 --policy task0_standard_frozen \
      --batch-size 2048 --outdir "$prep_out" \
      || die "Fit preprocessing member $member_id thất bại"
  done
  check_preprocessing
}

orchestrator_command() {
  "$PYTHON_BIN" src/training/fold_ensemble3_full.py \
    --config "$FULL_CONFIG" "${COMMON_P4[@]}" \
    --member-id 0 --member-id 1 --member-id 2 "$@"
}

latest_log_dir() {
  local pointer="$MONITOR_ROOT/latest_run.txt" log_dir
  [[ -s "$pointer" ]] || die "Chưa có lần launch nào."
  log_dir="$(<"$pointer")"
  [[ "$log_dir" == "$MONITOR_ROOT"/run_[0-9][0-9][0-9] ]] \
    || die "latest_run.txt chứa đường dẫn không hợp lệ."
  [[ -d "$log_dir" ]] || die "Không tìm thấy log directory: $log_dir"
  printf '%s\n' "$log_dir"
}

ensure_no_live_controller() {
  local pointer="$MONITOR_ROOT/latest_run.txt" previous pid_file pid
  if [[ -s "$pointer" ]]; then
    previous="$(<"$pointer")"
    pid_file="$previous/job.pid"
    if [[ -s "$pid_file" ]]; then
      pid="$(<"$pid_file")"
      if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        die "Controller PID $pid vẫn đang chạy: $previous"
      fi
    fi
  fi
}

next_log_dir() {
  local index name
  for index in $(seq 1 999); do
    printf -v name 'run_%03d' "$index"
    if [[ ! -e "$MONITOR_ROOT/$name" ]]; then
      printf '%s\n' "$MONITOR_ROOT/$name"
      return
    fi
  done
  die "Đã dùng hết namespace run_001..run_999"
}

start_run() {
  check_inputs
  check_preprocessing
  command -v nvidia-smi >/dev/null || die "Không tìm thấy nvidia-smi"
  command -v /usr/bin/time >/dev/null || die "Không tìm thấy /usr/bin/time"
  ensure_no_live_controller
  mkdir -p "$MONITOR_ROOT"
  local log_dir pid
  log_dir="$(next_log_dir)"
  mkdir "$log_dir" || die "Không tạo được log directory: $log_dir"
  printf '%s\n' "$log_dir" > "$MONITOR_ROOT/latest_run.txt"
  nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash -c '
      set -uo pipefail
      log_dir="$1"; repo_root="$2"; gpu_id="$3"
      python_bin="$4"; full_root="$5"; task_file="$6"; shift 6
      cd "$repo_root" || exit 1
      nvidia-smi --id="$gpu_id" \
        --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu,power.draw \
        --format=csv -l 2 > "$log_dir/gpu.csv" 2> "$log_dir/gpu_monitor.err" &
      monitor_pid=$!
      trap "kill ${monitor_pid} 2>/dev/null || true; wait ${monitor_pid} 2>/dev/null || true" EXIT
      /usr/bin/time -v "$@"
      status=$?
      if (( status == 0 )); then
        "$python_bin" src/eval/report.py \
          --full-root "$full_root" --task-file "$task_file" --overwrite
        status=$?
      fi
      printf "%s\n" "$status" > "$log_dir/exit_code.txt"
      exit "$status"
    ' p4-job "$log_dir" "$REPO_ROOT" "$GPU_ID" \
      "$PYTHON_BIN" "$FULL_ROOT" "$TASK_FILE" \
      "$PYTHON_BIN" src/training/fold_ensemble3_full.py \
      --config "$FULL_CONFIG" "${COMMON_P4[@]}" \
      --member-id 0 --member-id 1 --member-id 2 --execute \
      > "$log_dir/nohup.log" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "$pid" > "$log_dir/job.pid"
  echo "[STARTED] PID=$pid"
  echo "[LOG] $log_dir/nohup.log"
  echo "Theo dõi: bash $CONTROLLER_SCRIPT follow"
}

show_status() {
  local log_dir pid exit_code
  log_dir="$(latest_log_dir)"
  pid="$(<"$log_dir/job.pid")"
  echo "LOG_DIR=$log_dir"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    echo "[RUNNING] PID=$pid"
  elif [[ -s "$log_dir/exit_code.txt" ]]; then
    exit_code="$(<"$log_dir/exit_code.txt")"
    echo "[FINISHED] exit_code=$exit_code"
  else
    echo "[STOPPED/UNKNOWN] PID không sống và chưa có exit_code.txt"
  fi
  tail -n 40 "$log_dir/nohup.log"
}

follow_log() {
  local log_dir
  log_dir="$(latest_log_dir)"
  echo "[LOG] $log_dir/nohup.log"
  tail -f "$log_dir/nohup.log"
}

verify_results() {
  check_inputs
  check_preprocessing
  local missing=0 member_id task_id file
  for member_id in 0 1 2; do
    for task_id in 0 1 2 3 4 5 6 7; do
      for file in \
        "$FULL_ROOT/member_${member_id}/checkpoints/task_${task_id}.pt" \
        "$FULL_ROOT/member_${member_id}/task_${task_id}/completed_task.json" \
        "$FULL_ROOT/member_${member_id}/member_predictions/task_${task_id}/validation.npz" \
        "$FULL_ROOT/member_${member_id}/member_predictions/task_${task_id}/test.npz"; do
        [[ -s "$file" ]] || { echo "[MISSING] $file"; missing=$((missing + 1)); }
      done
    done
  done
  for task_id in 0 1 2 3 4 5 6 7; do
    for file in \
      "$FULL_ROOT/offline_evaluation/task_${task_id}/oof.npz" \
      "$FULL_ROOT/offline_evaluation/task_${task_id}/test.npz" \
      "$FULL_ROOT/offline_evaluation/task_${task_id}/frozen_threshold.json" \
      "$FULL_ROOT/offline_evaluation/task_${task_id}/oof_threshold_report.json" \
      "$FULL_ROOT/offline_evaluation/task_${task_id}/test_threshold_report.json"; do
      [[ -s "$file" ]] || { echo "[MISSING] $file"; missing=$((missing + 1)); }
    done
  done
  [[ -s "$FULL_ROOT/full_manifest.json" ]] \
    || { echo "[MISSING] $FULL_ROOT/full_manifest.json"; missing=$((missing + 1)); }
  (( missing == 0 )) || die "$missing artifact bắt buộc đang thiếu"
  orchestrator_command
  echo "[OK] P4 ensemble3 hoàn tất và artifacts đầy đủ"
}

build_final_report() {
  verify_results
  "$PYTHON_BIN" src/eval/report.py \
    --full-root "$FULL_ROOT" \
    --task-file "$TASK_FILE" \
    --overwrite
  echo "[OK] $FULL_ROOT/final_results/ensemble3_p4_final_report.md"
}

usage() {
  cat <<'EOF'
Usage: bash scripts/run_p4_ensemble3.sh ACTION

  check      Kiểm tra input, task hash và fold root.
  prepare    Tạo/skip scaler P4 cho member 0, 1, 2.
  dry-run    Hiện kế hoạch; không train và không ghi output.
  start      Nohup member 0 -> 1 -> 2 -> ensemble/UE; tự resume/skip.
  status     Xem PID, exit code và log cuối.
  follow     Theo dõi log; Ctrl+C chỉ thoát tail.
  verify     Kiểm tra artifacts sau khi toàn bộ run hoàn tất.
  report     Tạo lại bảng CSV và báo cáo Markdown cuối từ artifacts.

Optional: P4_GPU_ID=0, P4_PYTHON=python, P4_FOLD_ROOT=/absolute/path/to/folds
Advanced namespace overrides: P4_FULL_CONFIG, P4_FULL_ROOT, P4_MONITOR_ROOT.
EOF
}

action="${1:-}"
case "$action" in
  check) check_inputs ;;
  prepare) prepare_preprocessing ;;
  dry-run) check_inputs; check_preprocessing; orchestrator_command ;;
  start) start_run ;;
  status) show_status ;;
  follow) follow_log ;;
  verify) verify_results ;;
  report) build_final_report ;;
  help|-h|--help|"") usage ;;
  *) usage; die "Action không hợp lệ: $action" ;;
esac
