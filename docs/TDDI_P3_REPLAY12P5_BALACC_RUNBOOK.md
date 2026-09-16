# P3 replay 12,5% + threshold Balanced Accuracy

Run này giữ nguyên P3, folds, scaler B, backbone, loss và buffer 4%. Các thay đổi
so với baseline là:

1. Khi replay bị `repeat_cap=3` giới hạn, mỗi epoch dùng một current subset xoay
   vòng để replay thực tế vẫn bằng 12,5%. Current rows không bị loại khỏi toàn
   trajectory; task 7 dự kiến phủ hết current data sau khoảng ba epoch.
2. Threshold được chọn trên OOF bằng Balanced Accuracy lớn nhất với coverage tối
   thiểu 50%. Test chỉ load threshold đã đóng băng.
3. Số epoch tối đa tăng từ 20 lên 30; patience vẫn là 5. Đây là trần tối đa,
   không bắt buộc mọi task chạy đủ 30 epoch nếu validation đã ngừng cải thiện.

Run mới phải dùng namespace mới. Không resume checkpoint baseline vì sampler
contract đã khác. Threshold chỉ ảnh hưởng tập dự đoán được chọn; nó không thay đổi
Balanced Accuracy trên full test set.

## 1. Khai báo đường dẫn trên server

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
export FULL_CONFIG="$REPO_ROOT/configs/full_tddi_p3_fold_ensemble3_seed0_replay12p5.json"
export THRESHOLD_CONFIG="$REPO_ROOT/configs/eval_tddi_p3_ensemble_entropy_balanced_accuracy_threshold.json"
export FULL_ROOT="$REPO_ROOT/outputs/stratified_ensemble3/full_p3_seed0_replay12p5_rotating_v2"
export MONITOR_ROOT="$REPO_ROOT/outputs/stratified_ensemble3_monitor/full_p3_seed0_replay12p5_rotating_v2"
mkdir -p "$MONITOR_ROOT"
```

Nếu fold preparation trên server nằm ở timestamp khác, chỉ sửa `FOLD_ROOT` tới
folder chứa đúng `fold_assignments.parquet` và `fold_manifest.json` đã audit.

## 2. Tests và preflight

```bash
git pull --ff-only

python -m pytest \
  tests/test_fold_replay_sampler.py \
  tests/test_fold_pilot_training.py \
  tests/test_fold_replay_checkpoint.py \
  tests/test_fold_ensemble3_full.py \
  tests/test_confidence_threshold.py \
  -q

for FILE in \
  "$TRAIN" "$VALIDATION" "$TEST" "$FEATURES" "$TASK_FILE" \
  "$FOLD_ROOT/fold_assignments.parquet" "$FOLD_ROOT/fold_manifest.json" \
  "$FULL_CONFIG" "$THRESHOLD_CONFIG"
do
  test -s "$FILE" && echo "[OK] $FILE" || echo "[MISSING] $FILE"
done

for MEMBER_ID in 0 1 2
do
  test -s "$PREP_ROOT/member_${MEMBER_ID}/B/fold_preprocessing.json" \
    && echo "[OK] scaler member $MEMBER_ID" \
    || echo "[MISSING] scaler member $MEMBER_ID"
done
```

Không tiếp tục nếu test fail hoặc còn `[MISSING]`.

## 3. Dry-run

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

grep -q -- '--fold-replay-policy stratified_fraction_rotating_current_v2' \
  "$MONITOR_ROOT/dry_run.txt" \
  && echo '[OK] rotating-current v2' \
  || echo '[STOP] sai sampler policy'
```

Dry-run không train và không tạo model. Nó phải liệt kê member `0 → 1 → 2`, đủ
task 0–7, `--epochs 30`, `--patience 5` và policy
`stratified_fraction_rotating_current_v2`.

## 4. Hàm chạy từng member bằng nohup

```bash
launch_member() {
  local MEMBER_ID="$1"
  local LOG_DIR="$MONITOR_ROOT/member_${MEMBER_ID}"

  test ! -e "$LOG_DIR" || {
    echo "[STOP] $LOG_DIR đã tồn tại"
    return 1
  }
  mkdir -p "$LOG_DIR"

  nohup env \
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python src/training/fold_ensemble3_full.py \
      --config "$FULL_CONFIG" \
      "${COMMON_FULL[@]}" \
      --member-id "$MEMBER_ID" \
      --execute \
      > "$LOG_DIR/nohup.log" 2>&1 < /dev/null &

  echo $! | tee "$LOG_DIR/job.pid"
  echo "LOG_DIR=$LOG_DIR"
}
```

Chạy lần lượt, không chạy đồng thời:

```bash
launch_member 0
tail -f "$MONITOR_ROOT/member_0/nohup.log"
```

Sau khi member 0 kết thúc:

```bash
ps -fp "$(cat "$MONITOR_ROOT/member_0/job.pid")"
tail -n 100 "$MONITOR_ROOT/member_0/nohup.log"
test -s "$FULL_ROOT/member_0/checkpoints/task_7.pt" && echo '[OK] member 0'

python src/training/fold_ensemble3_full.py \
  --config "$FULL_CONFIG" "${COMMON_FULL[@]}" --member-id 0
```

Dry-run cuối phải báo member 0 `complete`. Sau đó làm tương tự:

```bash
launch_member 1
tail -f "$MONITOR_ROOT/member_1/nohup.log"

# Chỉ sau khi member 1 hoàn tất:
launch_member 2
tail -f "$MONITOR_ROOT/member_2/nohup.log"
```

Khi member 2 hoàn tất, orchestrator thấy đủ ba member và tự chạy offline OOF/test
ensemble, UE và threshold Balanced Accuracy cho task 0–7.

## 5. Kiểm tra sampler và kết quả

Kiểm tra task 7 của từng member:

```bash
for MEMBER_ID in 0 1 2
do
  python - "$FULL_ROOT/member_${MEMBER_ID}/task_7/epoch_1_audit.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))["sampling"]
print(sys.argv[1])
for key in ("policy", "current_count", "current_draws", "actual_replay_draws",
            "actual_fraction", "repeat_cap", "max_repeat", "current_coverage"):
    print(f"  {key}: {p.get(key)}")
PY
done
```

Acceptance chính:

- `policy = fold_fraction_capped_rotating_current_v2`;
- `actual_fraction = 0.125` khi capacity-limited;
- `max_repeat <= 3`;
- không NaN/OOM;
- đủ checkpoint và prediction artifacts task 0–7;
- threshold artifact có rule
  `max_balanced_accuracy_subject_to_min_coverage` và coverage tối thiểu 0,50.

Kết quả ensemble cuối nằm tại:

```text
outputs/stratified_ensemble3/full_p3_seed0_replay12p5_rotating_v2/offline_evaluation/task_7/
```
