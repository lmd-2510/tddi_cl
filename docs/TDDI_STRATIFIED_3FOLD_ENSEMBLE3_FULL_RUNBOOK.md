# Full P3 Ensemble3 — hướng dẫn chạy GPU

Study chính thức trong tài liệu này là:

```text
replay_distill_fixed_budget_uniform
× tddi_paper_member
× stratified 3-fold ensemble
× P3 tail-to-head
× experiment seed 0 / fold seed 42
× task 0–7 / 178 class
```

Ba trajectory phải chạy tuần tự `member 0 → member 1 → member 2`. Mỗi member dùng
scaler B riêng đã fit từ hai training folds và class task 0. Full run dùng namespace
mới, không tiếp tục trong output pilot task 0–1.

## 1. Vào repo và khai báo đường dẫn

```bash
cd "$HOME/DrugDrug/cil-tddi/cil-tddi"
conda activate ai_env

export REPO_ROOT="$PWD"
export TRAIN="$REPO_ROOT/train_extracted.parquet"
export VALIDATION="$REPO_ROOT/validation_extracted.parquet"
export TEST="$REPO_ROOT/test_extracted.parquet"
export FEATURES="$REPO_ROOT/study_assets/data_schema/feature_columns.json"
export TASK_FILE="$REPO_ROOT/study_assets/task_protocols/tail_to_head_tasks.json"
export FOLD_ROOT="$REPO_ROOT/outputs/fold_preparation_seed42_20260912_161002/folds"
export PREP_ROOT="$REPO_ROOT/study_assets/preprocessing_ab_seed0_fold42"
export THRESHOLD_CONFIG="$REPO_ROOT/configs/eval_tddi_p3_ensemble_entropy_threshold.json"
export FULL_CONFIG="$REPO_ROOT/configs/full_tddi_p3_fold_ensemble3_seed0.json"
export FULL_ROOT="$REPO_ROOT/outputs/stratified_ensemble3/full_p3_seed0_8tasks"
export MONITOR_ROOT="$REPO_ROOT/outputs/stratified_ensemble3_monitor/full_p3_seed0_8tasks"

mkdir -p "$MONITOR_ROOT"
```

Sau khi đóng SSH, phải khai báo lại các biến trên. Không đổi `FOLD_ROOT`, scaler,
task file, seed, policy hoặc hyperparameter giữa các member.

## 2. Cập nhật code và chạy tests

Chỉ pull khi worktree không chứa thay đổi code chưa lưu:

```bash
git status --short
git pull --ff-only

python -m pytest \
  tests/test_fold_ensemble3_full.py \
  tests/test_fold_ensemble3_pilot.py \
  tests/test_fold_pilot_training.py \
  tests/test_fold_replay_checkpoint.py \
  tests/test_stratified_ensemble_mode.py \
  tests/test_confidence_threshold.py \
  tests/test_build_final_report.py \
  -q
```

Không tiếp tục nếu test fail.

## 3. Kiểm tra input và tài nguyên

```bash
for file in \
  "$TRAIN" "$VALIDATION" "$TEST" "$FEATURES" "$TASK_FILE" \
  "$FOLD_ROOT/fold_assignments.parquet" \
  "$FOLD_ROOT/fold_manifest.json" \
  "$FULL_CONFIG" "$THRESHOLD_CONFIG"
do
  test -s "$file" && echo "[OK] $file" || echo "[MISSING] $file"
done

for MEMBER_ID in 0 1 2
do
  test -s "$PREP_ROOT/member_${MEMBER_ID}/B/fold_preprocessing.json" \
    && echo "[OK] scaler B member $MEMBER_ID" \
    || echo "[MISSING] scaler B member $MEMBER_ID"
done

sha256sum "$TASK_FILE" \
  "$FOLD_ROOT/fold_assignments.parquet" \
  "$FOLD_ROOT/fold_manifest.json" \
  "$FULL_CONFIG" "$THRESHOLD_CONFIG" \
  | tee "$MONITOR_ROOT/input_hashes.txt"

nvidia-smi
free -h
df -h "$REPO_ROOT"
```

Khuyến nghị trước khi bắt đầu: GPU CUDA 0 không có job lạ, ít nhất 8 GiB VRAM trống,
RAM available ít nhất 24 GiB và disk trống ít nhất 80 GiB. Pilot đo được peak khoảng
1.95 GiB tại task 1, nhưng task sau có nhiều dữ liệu và artifact hơn nên vẫn phải theo
dõi. Không tiếp tục nếu còn `[MISSING]`.

## 4. Dry-run full bắt buộc

```bash
COMMON_FULL=(
  --train "$TRAIN"
  --validation "$VALIDATION"
  --test "$TEST"
  --feature-cols "$FEATURES"
  --fold-assignments "$FOLD_ROOT/fold_assignments.parquet"
  --fold-manifest "$FOLD_ROOT/fold_manifest.json"
  --task-file "$TASK_FILE"
  --preprocessing-root "$PREP_ROOT"
  --threshold-config "$THRESHOLD_CONFIG"
  --output-root "$FULL_ROOT"
  --device cuda
)

python src/training/fold_ensemble3_full.py \
  --config "$FULL_CONFIG" \
  "${COMMON_FULL[@]}" \
  | tee "$MONITOR_ROOT/dry_run.txt"
```

Dry-run hợp lệ phải có:

- `full scope task 0-7`;
- ba member theo thứ tự `0 → 1 → 2`;
- seed `409845317`, `215626784`, `3041879697`;
- mỗi command có `--stop-after-task 7`;
- `--fold-replay-policy stratified_fraction_v1` có dấu cách;
- sample-normalized ranking, preprocessing B và export `validation test`;
- không có traceback và chưa tạo model.

Kiểm tra riêng flag dễ bị sao chép sai:

```bash
grep -q -- '--fold-replay-policy stratified_fraction_v1' "$MONITOR_ROOT/dry_run.txt" \
  && echo "[OK] replay flag" \
  || echo "[STOP] replay flag bị sai"
```

## 5. Hàm chạy một member bằng nohup trên CUDA 0

```bash
launch_full_member() {
  local MEMBER_ID="$1"
  local LABEL="${2:-first}"
  local LOG_DIR="$MONITOR_ROOT/member_${MEMBER_ID}_${LABEL}"

  test ! -e "$LOG_DIR" || {
    echo "[STOP] $LOG_DIR đã tồn tại; kiểm tra PID/log hoặc dùng LABEL khác khi resume."
    return 1
  }
  mkdir -p "$LOG_DIR"

  nohup env \
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash -c '
      set -uo pipefail
      log_dir="$1"
      shift
      nvidia-smi --id=0 \
        --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu,power.draw \
        --format=csv -l 2 > "$log_dir/gpu.csv" 2> "$log_dir/gpu_monitor.err" &
      monitor_pid=$!
      trap "kill ${monitor_pid} 2>/dev/null || true; wait ${monitor_pid} 2>/dev/null || true" EXIT
      /usr/bin/time -v python "$@"
      status=$?
      printf "%s\n" "$status" > "$log_dir/exit_code.txt"
      exit "$status"
    ' full-job "$LOG_DIR" \
      src/training/fold_ensemble3_full.py \
      --config "$FULL_CONFIG" \
      "${COMMON_FULL[@]}" \
      --member-id "$MEMBER_ID" \
      --execute \
    > "$LOG_DIR/nohup.log" 2>&1 < /dev/null &

  echo $! | tee "$LOG_DIR/job.pid"
  echo "LOG_DIR=$LOG_DIR"
}
```

Hàm chỉ tồn tại trong terminal hiện tại. Sau khi đăng nhập SSH lại, khai báo lại biến,
`COMMON_FULL` và hàm trước khi launch/resume.

## 6. Chạy member 0

```bash
launch_full_member 0 first
export CURRENT_MEMBER=0
export CURRENT_LOG="$MONITOR_ROOT/member_0_first"

ps -fp "$(cat "$CURRENT_LOG/job.pid")"
tail -f "$CURRENT_LOG/nohup.log"
```

Log chi tiết training cũng nằm ở:

```bash
tail -f "$FULL_ROOT/member_0/stdout.log"
```

`Ctrl+C` chỉ thoát `tail`, không dừng job nohup.

Kiểm tra trạng thái bất kỳ lúc nào:

```bash
JOB_PID="$(cat "$CURRENT_LOG/job.pid")"
if ps -p "$JOB_PID" >/dev/null; then
  echo "[RUNNING] member $CURRENT_MEMBER PID=$JOB_PID"
else
  echo "[FINISHED] member $CURRENT_MEMBER"
fi
```

Khi PID kết thúc, exit code phải là `0`:

```bash
cat "$CURRENT_LOG/exit_code.txt"
tail -n 150 "$CURRENT_LOG/nohup.log"

for TASK_ID in 0 1 2 3 4 5 6 7
do
  test -s "$FULL_ROOT/member_0/checkpoints/task_${TASK_ID}.pt" || echo "[MISSING] checkpoint task $TASK_ID"
  test -s "$FULL_ROOT/member_0/task_${TASK_ID}/completed_task.json" || echo "[MISSING] completion task $TASK_ID"
  test -s "$FULL_ROOT/member_0/member_predictions/task_${TASK_ID}/validation.npz" || echo "[MISSING] validation task $TASK_ID"
  test -s "$FULL_ROOT/member_0/member_predictions/task_${TASK_ID}/test.npz" || echo "[MISSING] test task $TASK_ID"
done

python src/training/fold_ensemble3_full.py \
  --config "$FULL_CONFIG" "${COMMON_FULL[@]}" --member-id 0
```

Chỉ chạy member 1 khi dry-run cuối báo member 0 `status=complete completed_task=7`,
exit code `0`, không OOM/NaN và không có `[MISSING]`.

## 7. Chạy member 1 rồi member 2

Member 1:

```bash
launch_full_member 1 first
export CURRENT_MEMBER=1
export CURRENT_LOG="$MONITOR_ROOT/member_1_first"
ps -fp "$(cat "$CURRENT_LOG/job.pid")"
tail -f "$CURRENT_LOG/nohup.log"
```

Sau khi PID member 1 kết thúc, lặp acceptance task 0–7 với path `member_1`, xác nhận
exit code `0` và dry-run báo `complete` trước khi chạy member 2.

Member 2:

```bash
launch_full_member 2 first
export CURRENT_MEMBER=2
export CURRENT_LOG="$MONITOR_ROOT/member_2_first"
ps -fp "$(cat "$CURRENT_LOG/job.pid")"
tail -f "$CURRENT_LOG/nohup.log"
```

Không chạy hai member đồng thời. Sau task 7 của member 2, cùng job sẽ tiếp tục tạo OOF,
common-test ensemble/UE và frozen threshold cho task 0–7. Phải chờ PID toàn bộ job kết
thúc; không kết luận hoàn tất ngay khi log báo member 2 task 7 complete.

## 8. Resume sau interruption

Trước tiên chắc chắn process cũ đã thực sự dừng:

```bash
pgrep -afu "$USER" '[f]old_ensemble3_full.py|[t]rain_cil.py'
```

Nếu đã có checkpoint task boundary hợp lệ, gọi lại đúng member với label log mới:

```bash
launch_full_member 1 resume1
export CURRENT_MEMBER=1
export CURRENT_LOG="$MONITOR_ROOT/member_1_resume1"
tail -f "$CURRENT_LOG/nohup.log"
```

Orchestrator tự chọn checkpoint mới nhất và thêm `--resume-fold-checkpoint`. Không tự
trỏ checkpoint, không xóa output, không đổi config/path/seed. Nếu job dừng trước khi có
checkpoint task 0, output dở phải được kiểm tra riêng; guard sẽ không ghi đè nó.

## 9. Kiểm tra ensemble, UE và threshold sau member 2

```bash
cat "$CURRENT_LOG/exit_code.txt"
test -s "$FULL_ROOT/full_manifest.json" \
  && echo "[OK] full study manifest" \
  || echo "[MISSING] full study manifest"

for TASK_ID in 0 1 2 3 4 5 6 7
do
  ROOT="$FULL_ROOT/offline_evaluation/task_${TASK_ID}"
  for NAME in oof.npz test.npz frozen_threshold.json oof_threshold_report.json test_threshold_report.json
  do
    test -s "$ROOT/$NAME" || echo "[MISSING] task $TASK_ID $NAME"
  done
done
```

Không có `[MISSING]` mới dựng báo cáo cuối:

```bash
python src/eval/report.py \
  --full-root "$FULL_ROOT" \
  --task-file "$TASK_FILE"
```

Kết quả Markdown:

```bash
cat "$FULL_ROOT/final_results/ensemble3_p3_final_report.md"
```

Nếu chạy lại report đã tồn tại và chủ động muốn thay đúng các derived report files,
thêm `--overwrite`; lệnh này không train và không inference.

## 10. Bundle nhẹ để gửi review

```bash
cd "$FULL_ROOT"
tar -czf ensemble3_full8_review.tar.gz \
  full_manifest.json \
  offline_evaluation/task_*/frozen_threshold.json \
  offline_evaluation/task_*/oof_threshold_report.json \
  offline_evaluation/task_*/test_threshold_report.json \
  member_*/run_config.json \
  member_*/stdout.log \
  member_*/task_*/completed_task.json \
  member_*/task_*/metrics.json \
  final_results

ls -lh ensemble3_full8_review.tar.gz
```

Bundle review không chứa checkpoint, model weights, replay features hoặc `.npz` lớn.
Giữ nguyên các artifact đó trên server để audit/resume.

