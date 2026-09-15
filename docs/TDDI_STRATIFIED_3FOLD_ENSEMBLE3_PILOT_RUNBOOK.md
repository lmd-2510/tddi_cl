# Runbook GPU — pilot T-DDI stratified 3-fold ensemble3, P3 task 0–1

Mục tiêu của pilot là kiểm tra trọn pipeline đã duyệt trên **task 0–1**:

```text
replay_distill_fixed_budget_uniform
× tddi_paper_member
× P3 tail-to-head / experiment seed 0 / fold seed 42
× member 0, 1, 2 tuần tự
× OOF threshold đã freeze → common-test ensemble/UE/report
```

Pilot dùng preprocessing B `task0_standard_frozen`, exemplar ranking
`raw_sample_normalized_class_mean_control_v1` và ba budget `9260/9259/9259`.
Nó **không phải full 8 task** và không đủ để kết luận hiệu quả tại task 7.

## 1. Cập nhật repo và khai báo path server

```bash
cd "$HOME/DrugDrug/cil-tddi/cil-tddi"
conda activate ai_env
git pull --ff-only

export REPO_ROOT="$PWD"
export TRAIN="$REPO_ROOT/train_extracted.parquet"
export VALIDATION="$REPO_ROOT/validation_extracted.parquet"
export TEST="$REPO_ROOT/test_extracted.parquet"
export FEATURES="$REPO_ROOT/study_assets/data_schema/feature_columns.json"
export TASK_FILE="$REPO_ROOT/study_assets/task_protocols/tail_to_head_tasks.json"

# Assignment đã audit trên server của lần chuẩn bị fold trước.
export FOLD_ROOT="$REPO_ROOT/outputs/fold_preparation_seed42_20260912_161002/folds"
export PREP_ROOT="$REPO_ROOT/study_assets/preprocessing_ab_seed0_fold42"
export PILOT_ROOT="$REPO_ROOT/outputs/stratified_ensemble3/pilot_p3_seed0_task01"
export MONITOR_ROOT="$REPO_ROOT/outputs/stratified_ensemble3_monitor/pilot_p3_seed0_task01"
export PILOT_CONFIG="$REPO_ROOT/configs/pilot_tddi_p3_fold_ensemble3_seed0.json"
export THRESHOLD_CONFIG="$REPO_ROOT/configs/eval_tddi_p3_ensemble_entropy_threshold.json"
mkdir -p "$MONITOR_ROOT"
```

Không đổi `FOLD_ROOT` nếu vẫn dùng đúng assignment đã audit. Nếu server lưu artifact
ở path khác, chỉ sửa biến path; không dựng lại fold.

## 2. Kiểm tra input và chuẩn bị scaler riêng cho ba member

```bash
for file in \
  "$TRAIN" "$VALIDATION" "$TEST" "$FEATURES" "$TASK_FILE" \
  "$FOLD_ROOT/fold_assignments.parquet" \
  "$FOLD_ROOT/fold_manifest.json" \
  "$PILOT_CONFIG" "$THRESHOLD_CONFIG"
do
  test -s "$file" && echo "[OK] $file" || echo "[MISSING] $file"
done

sha256sum "$TASK_FILE" "$FOLD_ROOT/fold_assignments.parquet" \
  "$FOLD_ROOT/fold_manifest.json" | tee "$MONITOR_ROOT/input_hashes.txt"
```

Mỗi member phải có scaler B riêng, fit chỉ trên hai training folds và class task 0
của chính member. Lệnh sau **skip file đã có**, không ghi đè:

```bash
for MEMBER_ID in 0 1 2
do
  PREP_OUT="$PREP_ROOT/member_${MEMBER_ID}/B"
  if test -s "$PREP_OUT/fold_preprocessing.json"; then
    echo "[SKIP] scaler member $MEMBER_ID đã tồn tại"
    continue
  fi
  python scripts/prepare_fold_preprocessing.py \
    --train "$TRAIN" \
    --validation "$VALIDATION" \
    --test "$TEST" \
    --feature-cols "$FEATURES" \
    --assignments "$FOLD_ROOT/fold_assignments.parquet" \
    --manifest "$FOLD_ROOT/fold_manifest.json" \
    --task-file "$TASK_FILE" \
    --member-id "$MEMBER_ID" \
    --validation-fold "$MEMBER_ID" \
    --experiment-seed 0 \
    --fold-seed 42 \
    --policy task0_standard_frozen \
    --batch-size 2048 \
    --outdir "$PREP_OUT"
done

for MEMBER_ID in 0 1 2
do
  test -s "$PREP_ROOT/member_${MEMBER_ID}/B/fold_preprocessing.json" \
    && echo "[OK] preprocessing member $MEMBER_ID" \
    || echo "[MISSING] preprocessing member $MEMBER_ID"
done
```

Không tiếp tục nếu còn `[MISSING]`.

## 3. Kiểm tra máy và unit/synthetic tests

```bash
which python
python --version
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info(0)
    print("free_GiB:", free / 2**30, "total_GiB:", total / 2**30)
PY

nvidia-smi
free -h
df -h .

python -m pytest \
  tests/test_fold_ensemble3_pilot.py \
  tests/test_fold_pilot_training.py \
  tests/test_fold_replay_checkpoint.py \
  tests/test_stratified_ensemble_mode.py \
  tests/test_confidence_threshold.py \
  -q
```

Điều kiện tối thiểu: CUDA hoạt động, GPU không có job lạ chiếm gần hết VRAM, disk
còn đủ cho ba checkpoint/prediction artifact, tests đều pass.

## 4. Dry-run bắt buộc

```bash
COMMON=(
  --train "$TRAIN"
  --validation "$VALIDATION"
  --test "$TEST"
  --feature-cols "$FEATURES"
  --fold-assignments "$FOLD_ROOT/fold_assignments.parquet"
  --fold-manifest "$FOLD_ROOT/fold_manifest.json"
  --task-file "$TASK_FILE"
  --preprocessing-root "$PREP_ROOT"
  --threshold-config "$THRESHOLD_CONFIG"
  --output-root "$PILOT_ROOT"
  --device cuda
)

python src/training/fold_ensemble3_pilot.py \
  --config "$PILOT_CONFIG" \
  "${COMMON[@]}" \
  | tee "$MONITOR_ROOT/dry_run.txt"
```

Dry-run đúng phải hiện:

- member theo thứ tự `0 → 1 → 2` với seed `409845317`, `215626784`, `3041879697`;
- `--stop-after-task 1`, 20 epoch, preprocessing B và sample-normalized ranking;
- export `validation test`;
- evaluation đang chờ đủ ba member;
- không có lệnh full task 2–7 và không tạo model.

## 5. Hàm chạy nohup ổn định trên CUDA 0

Hàm này dùng tên thư mục rõ ràng, không sinh hậu tố ngẫu nhiên. `LABEL` cho phép giữ
log lần resume mà không xóa log cũ.

```bash
launch_member() {
  local MEMBER_ID="$1"
  local LABEL="${2:-first}"
  local LOG_DIR="$MONITOR_ROOT/member_${MEMBER_ID}_${LABEL}"

  test ! -e "$LOG_DIR" || {
    echo "[STOP] $LOG_DIR đã tồn tại; chọn LABEL khác hoặc kiểm tra PID/log."
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
    ' pilot-job "$LOG_DIR" \
      src/training/fold_ensemble3_pilot.py \
      --config "$PILOT_CONFIG" \
      "${COMMON[@]}" \
      --member-id "$MEMBER_ID" \
      --execute \
    > "$LOG_DIR/nohup.log" 2>&1 < /dev/null &

  echo $! | tee "$LOG_DIR/job.pid"
  echo "LOG_DIR=$LOG_DIR"
}
```

## 6. Chạy member 0, kiểm tra rồi mới chuyển member 1/2

```bash
launch_member 0 first
export CURRENT_LOG="$MONITOR_ROOT/member_0_first"
ps -fp "$(cat "$CURRENT_LOG/job.pid")"
tail -f "$CURRENT_LOG/nohup.log"
```

Khi PID kết thúc:

```bash
cat "$CURRENT_LOG/exit_code.txt"
tail -n 100 "$CURRENT_LOG/nohup.log"

for TASK_ID in 0 1
do
  test -s "$PILOT_ROOT/member_0/checkpoints/task_${TASK_ID}.pt" || echo "[MISSING] checkpoint task $TASK_ID"
  test -s "$PILOT_ROOT/member_0/task_${TASK_ID}/completed_task.json" || echo "[MISSING] completion task $TASK_ID"
  test -s "$PILOT_ROOT/member_0/member_predictions/task_${TASK_ID}/validation.npz" || echo "[MISSING] validation task $TASK_ID"
  test -s "$PILOT_ROOT/member_0/member_predictions/task_${TASK_ID}/test.npz" || echo "[MISSING] test task $TASK_ID"
done

python src/training/fold_ensemble3_pilot.py \
  --config "$PILOT_CONFIG" "${COMMON[@]}" --member-id 0
```

Dry-run cuối phải báo member 0 `complete`, task 1. Chỉ khi exit code `0`, loss hữu
hạn, không OOM và bốn prediction artifact tồn tại mới chạy tiếp:

```bash
launch_member 1 first
export CURRENT_MEMBER=1
export CURRENT_LOG="$MONITOR_ROOT/member_1_first"
ps -fp "$(cat "$CURRENT_LOG/job.pid")"
tail -f "$CURRENT_LOG/nohup.log"

# Sau khi PID kết thúc: exit code phải là 0, rồi kiểm tra artifact member_1.
cat "$CURRENT_LOG/exit_code.txt"
for TASK_ID in 0 1; do
  test -s "$PILOT_ROOT/member_${CURRENT_MEMBER}/checkpoints/task_${TASK_ID}.pt" || echo "[MISSING] checkpoint task $TASK_ID"
  test -s "$PILOT_ROOT/member_${CURRENT_MEMBER}/task_${TASK_ID}/completed_task.json" || echo "[MISSING] completion task $TASK_ID"
  test -s "$PILOT_ROOT/member_${CURRENT_MEMBER}/member_predictions/task_${TASK_ID}/validation.npz" || echo "[MISSING] validation task $TASK_ID"
  test -s "$PILOT_ROOT/member_${CURRENT_MEMBER}/member_predictions/task_${TASK_ID}/test.npz" || echo "[MISSING] test task $TASK_ID"
done

launch_member 2 first
export CURRENT_MEMBER=2
export CURRENT_LOG="$MONITOR_ROOT/member_2_first"
ps -fp "$(cat "$CURRENT_LOG/job.pid")"
tail -f "$CURRENT_LOG/nohup.log"

# Sau khi PID của member 2 kết thúc: exit code phải là 0. Job này còn chạy
# ensemble/threshold sau task 1, nên phải đợi toàn bộ PID kết thúc.
cat "$CURRENT_LOG/exit_code.txt"
test -s "$PILOT_ROOT/pilot_manifest.json" && echo "[OK] pilot manifest"
```

Không chạy hai lệnh `launch_member` cùng lúc. Lần member 2 hoàn thành sẽ tự chạy
OOF, common-test ensemble/UE và threshold/report vì lúc đó đủ ba member.

## 7. Resume sau interruption

Kiểm tra trước:

```bash
pgrep -afu "$USER" '[f]old_ensemble3_pilot.py|[t]rain_cil.py'
```

Nếu không còn process và checkpoint task boundary hợp lệ, chạy lại cùng member với
label mới; orchestrator tự thêm `--resume-fold-checkpoint`:

```bash
launch_member 1 resume1
```

Không xóa output member, không đổi config/path/seed và không tự trỏ checkpoint bằng
tay. Nếu dừng giữa task, attempt dở được giữ; resume bắt đầu từ boundary gần nhất.

## 8. Kiểm tra kết quả OOF, ensemble test và threshold

```bash
test -s "$PILOT_ROOT/pilot_manifest.json" && echo "[OK] pilot manifest"

for TASK_ID in 0 1
do
  EVAL_ROOT="$PILOT_ROOT/offline_evaluation/task_${TASK_ID}"
  for file in \
    oof.npz test.npz frozen_threshold.json \
    oof_threshold_report.json test_threshold_report.json
  do
    test -s "$EVAL_ROOT/$file" \
      && echo "[OK] task=$TASK_ID $file" \
      || echo "[MISSING] task=$TASK_ID $file"
  done
done

python - <<'PY'
import json
from pathlib import Path
import numpy as np

root = Path("outputs/stratified_ensemble3/pilot_p3_seed0_task01/offline_evaluation")
for task in (0, 1):
    frozen = json.loads((root / f"task_{task}/frozen_threshold.json").read_text())
    test = json.loads((root / f"task_{task}/test_threshold_report.json").read_text())
    with np.load(root / f"task_{task}/oof.npz", allow_pickle=False) as oof:
        oof_counts = np.unique(oof["prediction_count"]).tolist()
    with np.load(root / f"task_{task}/test.npz", allow_pickle=False) as ensemble:
        test_counts = np.unique(ensemble["prediction_count"]).tolist()
        unavailable = ensemble["unavailable_metrics"].tolist()
    print({
        "task": task,
        "threshold_status": frozen["selection_status"],
        "threshold": frozen["selected_threshold"],
        "target_met": frozen["target_met"],
        "test_full_accuracy": test["full_set"]["accuracy"],
        "test_selected_accuracy": test["threshold_score_selective_metrics"]["accuracy"],
        "test_coverage": test["threshold_score_selective_metrics"]["coverage"],
        "oof_prediction_count": oof_counts,
        "test_prediction_count": test_counts,
        "test_unavailable_ue": unavailable,
    })
PY
```

Kỳ vọng: OOF có `prediction_count=[1]`; test có `[3]`; test không thiếu MI/variance/
disagreement. `fallback` hoặc `no_selection` là kết quả hợp lệ, không được sửa
threshold bằng test. OOF đạt 95% cũng không bảo đảm selected test đạt 95%.

## 9. Acceptance và bundle gửi đánh giá

Go kỹ thuật khi:

- cả ba member hoàn tất task 1, đúng class head 38 rồi 58;
- checkpoint, run config, prediction artifacts và manifest đều có;
- không overlap train/held-out, provenance/fold/class alignment pass;
- task 1 có replay, repeat cap ≤3, loss hữu hạn;
- OOF đủ coverage một prediction/mẫu; test đủ ba prediction/mẫu;
- threshold ở đúng một trạng thái `target_met`, `fallback`, `no_selection`;
- không có test-driven refit hoặc artifact full task 2–7.

Tạo bundle nhẹ, không chứa model/checkpoint/Parquet/prediction `.npz`:

```bash
export REVIEW_DIR="$PILOT_ROOT/review_bundle"
test ! -e "$REVIEW_DIR" || {
  echo "[STOP] review_bundle đã tồn tại"
  false
}
mkdir -p "$REVIEW_DIR"

cp "$PILOT_CONFIG" "$THRESHOLD_CONFIG" "$PILOT_ROOT/pilot_manifest.json" "$REVIEW_DIR/"
for MEMBER_ID in 0 1 2
do
  mkdir -p "$REVIEW_DIR/member_${MEMBER_ID}"
  cp "$PILOT_ROOT/member_${MEMBER_ID}/run_config.json" "$REVIEW_DIR/member_${MEMBER_ID}/"
  cp "$PILOT_ROOT/member_${MEMBER_ID}/stdout.log" "$REVIEW_DIR/member_${MEMBER_ID}/"
  cp "$PILOT_ROOT/member_${MEMBER_ID}/task_0/metrics.json" "$REVIEW_DIR/member_${MEMBER_ID}/task0_metrics.json"
  cp "$PILOT_ROOT/member_${MEMBER_ID}/task_1/metrics.json" "$REVIEW_DIR/member_${MEMBER_ID}/task1_metrics.json"
  cp "$PILOT_ROOT/member_${MEMBER_ID}/task_0/completed_task.json" "$REVIEW_DIR/member_${MEMBER_ID}/task0_completed.json"
  cp "$PILOT_ROOT/member_${MEMBER_ID}/task_1/completed_task.json" "$REVIEW_DIR/member_${MEMBER_ID}/task1_completed.json"
done
for TASK_ID in 0 1
do
  mkdir -p "$REVIEW_DIR/offline_evaluation/task_${TASK_ID}"
  cp "$PILOT_ROOT/offline_evaluation/task_${TASK_ID}/frozen_threshold.json" \
    "$PILOT_ROOT/offline_evaluation/task_${TASK_ID}/oof_threshold_report.json" \
    "$PILOT_ROOT/offline_evaluation/task_${TASK_ID}/test_threshold_report.json" \
    "$REVIEW_DIR/offline_evaluation/task_${TASK_ID}/"
done

tar -C "$PILOT_ROOT" -czf "$PILOT_ROOT/ensemble3_task01_review.tar.gz" review_bundle
ls -lh "$PILOT_ROOT/ensemble3_task01_review.tar.gz"
```

Gửi `ensemble3_task01_review.tar.gz` để đánh giá go/no-go. **Dừng tại đây**; không
tự chạy full tám task hoặc đổi method/hyperparameter.
