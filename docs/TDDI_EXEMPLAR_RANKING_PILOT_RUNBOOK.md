# Runbook GPU — pilot không gian chọn exemplar

Mục tiêu: với preprocessing B đã duyệt (`task0_standard_frozen`), so sánh đúng hai
ranking sau trên member 0, P3 task 0–1, validation-only:

- `sample_normalized`: chuẩn hóa độc lập từng mẫu rồi chọn gần class mean;
- `pipeline_input`: dùng descriptor sau frozen StandardScaler B, trước LayerNorm,
  rồi chọn gần class mean.

Hai case giữ nguyên quota, replay fraction/cap, sampler, model và hyperparameter.
Exemplar IDs **được phép khác** vì đó chính là biến đang kiểm tra. Không chạy test,
ensemble, threshold hoặc full8 trong runbook này.

## 1. Cập nhật code và kiểm tra môi trường

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

# Sửa đúng hai path dưới theo artifact server đã audit/fit trước đó.
export FOLD_ROOT="$REPO_ROOT/outputs/fold_preparation_seed42_20260912_161002/folds"
export PREP_ROOT="$REPO_ROOT/study_assets/preprocessing_ab_seed0_fold42"

export RANK_ROOT="$REPO_ROOT/outputs/exemplar_ranking_v1"
export MONITOR_ROOT="$REPO_ROOT/outputs/exemplar_ranking_monitor_v1"
mkdir -p "$MONITOR_ROOT"
```

`PREP_ROOT` phải có cấu trúc
`member_0/B/fold_preprocessing.json`. Không dùng artifact A và không fit scaler lại.

```bash
for file in \
  "$TRAIN" "$VALIDATION" "$TEST" "$FEATURES" "$TASK_FILE" \
  "$FOLD_ROOT/fold_assignments.parquet" \
  "$FOLD_ROOT/fold_manifest.json" \
  "$PREP_ROOT/member_0/B/fold_preprocessing.json"
do
  test -s "$file" && echo "[OK] $file" || echo "[MISSING] $file"
done

nvidia-smi
df -h .
free -h
```

Không tiếp tục nếu còn `[MISSING]`.

## 2. Chạy unit/synthetic tests

```bash
python -m pytest \
  tests/test_fold_replay_buffer.py \
  tests/test_fold_pilot_training.py \
  tests/test_fold_replay_checkpoint.py \
  tests/test_fold_ab_study.py \
  tests/test_compare_preprocessing_pilots.py \
  tests/test_exemplar_ranking_pilot.py \
  -q
```

## 3. Khai báo đối số server và dry-run smoke

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
  --output-root "$RANK_ROOT"
  --device cuda
  --member-id 0
)

python src/training/fold_ab_study.py \
  --config \
    configs/smoke_tddi_p3_fold_ranking_sample_normalized_seed0.json \
    configs/smoke_tddi_p3_fold_ranking_pipeline_input_seed0.json \
  "${COMMON[@]}"
```

Dry-run hợp lệ phải in theo thứ tự `sample_normalized -> pipeline_input`, cùng
member seed `409845317`, budget `9260`, preprocessing
`task0_standard_frozen`, task 0–1 và `validation-only`. Dry-run không tạo model.

## 4. Chạy smoke tuần tự bằng nohup trên CUDA 0

Một process điều phối sẽ đợi case đầu kết thúc rồi mới chạy case thứ hai; không giữ
hai model trên GPU cùng lúc.

```bash
export SMOKE_LOG_DIR="$MONITOR_ROOT/smoke_ranking_pair_v1"
test ! -e "$SMOKE_LOG_DIR" || {
  echo "[STOP] $SMOKE_LOG_DIR đã tồn tại; kiểm tra PID/log trước khi chạy lại."
  false
}
mkdir -p "$SMOKE_LOG_DIR"

nohup env \
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python src/training/fold_ab_study.py \
    --config \
      configs/smoke_tddi_p3_fold_ranking_sample_normalized_seed0.json \
      configs/smoke_tddi_p3_fold_ranking_pipeline_input_seed0.json \
    "${COMMON[@]}" \
    --execute \
  > "$SMOKE_LOG_DIR/nohup.log" 2>&1 < /dev/null &

echo $! | tee "$SMOKE_LOG_DIR/job.pid"
```

Theo dõi:

```bash
SMOKE_PID="$(cat "$SMOKE_LOG_DIR/job.pid")"
ps -fp "$SMOKE_PID"
tail -f "$SMOKE_LOG_DIR/nohup.log"

tail -n 100 "$RANK_ROOT/smoke/sample_normalized/member_0/stdout.log"
tail -n 100 "$RANK_ROOT/smoke/pipeline_input/member_0/stdout.log"
```

Khi process kết thúc:

```bash
wait "$SMOKE_PID" 2>/dev/null; echo "exit=$?"

python src/training/fold_ab_study.py \
  --config \
    configs/smoke_tddi_p3_fold_ranking_sample_normalized_seed0.json \
    configs/smoke_tddi_p3_fold_ranking_pipeline_input_seed0.json \
  "${COMMON[@]}"
```

Dry-run sau smoke phải báo cả hai case `complete`, completed task 1. Nếu interrupted
sau task boundary, chạy lại đúng lệnh `nohup`; orchestrator sẽ resume checkpoint hợp
lệ và không ghi đè task đã hoàn tất. Nếu dừng giữa task, attempt dở được giữ lại.

## 5. Báo cáo smoke và tiêu chí go kỹ thuật

```bash
export SMOKE_REPORT="$RANK_ROOT/reviews/smoke_$(date -u +%Y%m%dT%H%M%SZ)"
python scripts/compare_exemplar_ranking_pilots.py \
  --sample-normalized-run "$RANK_ROOT/smoke/sample_normalized/member_0" \
  --pipeline-input-run "$RANK_ROOT/smoke/pipeline_input/member_0" \
  --task-file "$TASK_FILE" \
  --outdir "$SMOKE_REPORT"

cat "$SMOKE_REPORT/comparison.md"
```

Go kỹ thuật khi:

- alignment `PASS`, không có test metric;
- cả hai hoàn tất task 1, loss hữu hạn, không OOM;
- cùng current/validation IDs, class map và quota/raw-label allocation;
- replay fraction/cap hợp lệ;
- ranking policy và preprocessing SHA256 xuất hiện trong run config/checkpoint;
- exemplar IDs có thể giống hoặc khác, không dùng điều đó làm điều kiện fail.

Smoke 3 epoch không dùng để chọn ranking.

## 6. Chạy pilot 20 epoch tuần tự

Chỉ chạy sau khi smoke đạt go kỹ thuật. Namespace pilot không dùng lại smoke.

```bash
export PILOT_LOG_DIR="$MONITOR_ROOT/pilot_ranking_pair_v1"
test ! -e "$PILOT_LOG_DIR" || {
  echo "[STOP] $PILOT_LOG_DIR đã tồn tại; kiểm tra PID/checkpoint trước."
  false
}
mkdir -p "$PILOT_LOG_DIR"

nohup env \
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python src/training/fold_ab_study.py \
    --config \
      configs/pilot_tddi_p3_fold_ranking_sample_normalized_seed0.json \
      configs/pilot_tddi_p3_fold_ranking_pipeline_input_seed0.json \
    "${COMMON[@]}" \
    --execute \
  > "$PILOT_LOG_DIR/nohup.log" 2>&1 < /dev/null &

echo $! | tee "$PILOT_LOG_DIR/job.pid"
tail -f "$PILOT_LOG_DIR/nohup.log"
```

Kiểm tra hoàn tất:

```bash
PILOT_PID="$(cat "$PILOT_LOG_DIR/job.pid")"
ps -fp "$PILOT_PID" || true

python src/training/fold_ab_study.py \
  --config \
    configs/pilot_tddi_p3_fold_ranking_sample_normalized_seed0.json \
    configs/pilot_tddi_p3_fold_ranking_pipeline_input_seed0.json \
  "${COMMON[@]}"
```

## 7. Tạo comparison và bundle gửi đánh giá

```bash
export PILOT_REPORT="$RANK_ROOT/reviews/pilot_$(date -u +%Y%m%dT%H%M%SZ)"
python scripts/compare_exemplar_ranking_pilots.py \
  --sample-normalized-run "$RANK_ROOT/pilot/sample_normalized/member_0" \
  --pipeline-input-run "$RANK_ROOT/pilot/pipeline_input/member_0" \
  --task-file "$TASK_FILE" \
  --outdir "$PILOT_REPORT" \
  --bundle

cat "$PILOT_REPORT/comparison.md"
ls -lh "$PILOT_REPORT/ranking_review.tar.gz"
```

Gửi file `ranking_review.tar.gz` để đánh giá. Bundle loại model, checkpoint,
Parquet và scaler nặng; giữ audit/config/metric/log cần cho phép so sánh.

**Dừng sau bước này.** Không tự chọn ranking, không chạy member 1/2, test,
ensemble hoặc full8 trước khi người dùng duyệt không gian exemplar cuối cùng.

