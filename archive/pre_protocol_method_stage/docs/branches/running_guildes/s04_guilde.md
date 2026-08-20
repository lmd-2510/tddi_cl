# Hướng dẫn chạy S04 fixed-budget replay

S04 thêm baseline `replay_distill_fixed_budget_uniform`. Baseline này khóa độc lập
hai đại lượng:

- storage: tối đa 6.800 exemplar unique trên toàn bộ old/current classes đã học;
- exposure: đúng 6.800 old-sample draws trong mỗi epoch từ task 1.

Task 0 chỉ dùng current samples. Từ task 1, mỗi current sample xuất hiện đúng một
lần mỗi epoch; replay quota được chia đều theo raw class ID, phần dư luân phiên theo
epoch. Validation và test không được dùng để tạo buffer hoặc quota.

## CPU smoke 8 tasks

Smoke dùng tám lớp support cao, MLP-small và một epoch nhưng vẫn giữ budget
`6800/6800`. Giới hạn 900 train rows/task làm memory đạt đúng 6.800 ở task 7.

```bash
SMOKE_ROOT=$(mktemp -d /private/tmp/ddi-s04-smoke.XXXXXX)

.venv/bin/python src/training/train_cil.py \
  --train train_extracted.parquet \
  --validation validation_extracted.parquet \
  --test test_extracted.parquet \
  --feature-cols outputs/audit/feature_columns.json \
  --scaler outputs/preprocess/scaler.pkl \
  --task-file configs/s04_cpu_smoke_tasks.json \
  --outdir "${SMOKE_ROOT}/run" \
  --method replay_distill_fixed_budget_uniform \
  --variant small \
  --batch-size 1024 \
  --epochs 1 \
  --patience 1 \
  --seed 0 \
  --device cpu \
  --total-memory-budget 6800 \
  --replay-draws-per-epoch 6800 \
  --max-train-rows-per-task 900 \
  --max-validation-rows-per-task 128 \
  --max-test-rows-per-task 128 \
  --export-s02
```

Audit smoke bằng chính validator O07:

```bash
.venv/bin/python src/eval/run_s04_aggregation.py \
  --runs-root "${SMOKE_ROOT}" \
  --outdir "${SMOKE_ROOT}/aggregate" \
  --expected-seeds 0 \
  --expected-tasks 8 \
  --total-memory-budget 6800 \
  --replay-draws-per-epoch 6800
```

## Full MPS từ clean worktree

Trước khi chạy, commit implementation và tạo detached worktree tại đúng commit đó.
Không copy/stash các thay đổi dirty từ worktree chính. Script từ worktree clean nhận
`DATA_ROOT` và `OUTPUT_ROOT` tuyệt đối, kiểm tra MPS và từ chối chạy khi còn dưới
35 GiB.

```bash
S04_COMMIT=$(git rev-parse HEAD)
S04_WORKTREE=$(mktemp -d /private/tmp/ddi-s04-worktree.XXXXXX)
git worktree add --detach "${S04_WORKTREE}" "${S04_COMMIT}"

DATA_ROOT=$(pwd)
OUTPUT_ROOT="${DATA_ROOT}/outputs/runs_s04"
PYTHON_BIN="${DATA_ROOT}/.venv/bin/python"

cd "${S04_WORKTREE}"
./scripts/train_s04_mps.sh 0 "${DATA_ROOT}" "${OUTPUT_ROOT}" "${PYTHON_BIN}"
```

Sau seed 0, chạy partial validator từ worktree clean. Chỉ chạy tiếp seeds 1–4 nếu
O06, O01/O02 và S02 manifest pass.

```bash
"${PYTHON_BIN}" src/eval/run_s04_aggregation.py \
  --runs-root "${OUTPUT_ROOT}" \
  --outdir "${S04_WORKTREE}/seed0_audit" \
  --expected-seeds 0 1 2 3 4 \
  --allow-partial

for seed in 1 2 3 4; do
  ./scripts/train_s04_mps.sh "${seed}" "${DATA_ROOT}" "${OUTPUT_ROOT}" "${PYTHON_BIN}"
done
```

## O07 và calibration S03

```bash
"${PYTHON_BIN}" src/eval/run_s04_aggregation.py \
  --runs-root "${OUTPUT_ROOT}" \
  --outdir "${DATA_ROOT}/outputs/s04"

"${PYTHON_BIN}" src/eval/run_s03_calibration.py \
  --runs-root "${OUTPUT_ROOT}" \
  --outdir "${DATA_ROOT}/outputs/s04/calibration"
```

Mỗi run phải có 8 task, một O06 không trùng khóa và 16 S02 exports. O07 phải có
đúng năm hàng seeds 0–4. So sánh với `replay_distill` legacy phải ghi rõ legacy dùng
per-class cap 50 và inverse-frequency replacement sampling, nên không cùng storage
hoặc replay-exposure budget.
