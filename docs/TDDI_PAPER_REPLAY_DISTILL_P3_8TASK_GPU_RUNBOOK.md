# Full GPU runbook — T-DDI Ensemble3 Replay-Distill P3 seed 0

## 1. Study contract

```text
method          = replay_distill_fixed_budget_uniform
backbone        = tddi_paper_member
architecture    = LayerNorm(3780) -> 7560 -> 7560 -> expandable head
protocol        = P3 tail_to_head
experiment seed = 0
member IDs      = 0, 1, 2
tasks           = 8, layout [38,20,20,20,20,20,20,20]
```

Config:

```text
configs/tddi_ensemble3_replay_distill_p3_seed0.json
```

Output:

```text
outputs/full/tddi_ensemble3_replay_distill_p3_seed0_8tasks_v1
```

Không có legacy T-DDI backbone trong study này. Không lệnh nào tự chạy khi clone repo;
training chỉ bắt đầu khi người dùng chạy lệnh có `--execute`.

## 2. Vào repo và kích hoạt môi trường

```bash
cd "$HOME/DrugDrug/cil-tddi/cil-tddi"
conda activate ai_env

which python
python --version
git status --short
```

## 3. Kiểm tra data, P3 và tài nguyên

```bash
for file in \
  train_extracted.parquet \
  validation_extracted.parquet \
  test_extracted.parquet \
  outputs/audit/feature_columns.json \
  outputs/preprocess/scaler.pkl \
  outputs/tasks/tail_to_head_tasks.json
do
  test -s "$file" || { echo "[MISSING] $file"; exit 1; }
  echo "[OK] $file"
done
```

Xác minh task-file:

```bash
mkdir -p outputs/remote_preflight

python - <<'PY' | tee outputs/remote_preflight/p3_task_validation.txt
import hashlib
import json
from pathlib import Path

path = Path("outputs/tasks/tail_to_head_tasks.json")
payload = json.loads(path.read_text(encoding="utf-8"))
tasks = payload["tasks"]
layout = [len(task["classes"]) for task in tasks]
classes = [int(value) for task in tasks for value in task["classes"]]

print("protocol:", payload["protocol"])
print("task seed:", payload.get("seed"))
print("task count:", len(tasks))
print("layout:", layout)
print("unique classes:", len(set(classes)))
print("sha256:", hashlib.sha256(path.read_bytes()).hexdigest())

assert payload["protocol"] == "tail_to_head"
assert payload.get("seed") is None
assert layout == [38, 20, 20, 20, 20, 20, 20, 20]
assert len(classes) == len(set(classes)) == 178
PY

sha256sum outputs/tasks/tail_to_head_tasks.json \
  | tee outputs/remote_preflight/p3_task_sha256.txt
```

`seed: null` là đúng: P3 sắp lớp theo tail-to-head deterministically, không dùng random
task permutation. Experiment seed 0 vẫn điều khiển shared study metadata, replay
exemplar và member-seed derivation.

Kiểm tra máy:

```bash
nvidia-smi
free -h
df -h .

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
free_bytes, total_bytes = torch.cuda.mem_get_info(0)
print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
print("free GiB:", free_bytes / 2**30)
print("total GiB:", total_bytes / 2**30)
PY
```

Khuyến nghị trước run: CUDA 0 trống khoảng 20 GiB, RAM available từ 32 GiB và disk
trống từ 50 GiB. Không chạy model khác trên CUDA 0.

## 4. Chạy unit tests và dry-run

```bash
python -m pytest \
  tests/test_tddi_paper_member.py \
  tests/test_ensemble_seed.py \
  tests/test_replay_checkpoint.py \
  tests/test_tddi_ensemble3_replay_study.py \
  tests/test_member_prediction_export.py \
  tests/test_offline_ensemble.py \
  tests/test_offline_ue_audit.py \
  tests/test_confidence_threshold.py \
  tests/test_offline_temperature_calibration.py \
  -q
```

Dry-run tất cả member:

```bash
python src/training/tddi_ensemble3_study.py \
  --config configs/tddi_ensemble3_replay_distill_p3_seed0.json \
  | tee outputs/remote_preflight/p3_full_dry_run.txt
```

Dry-run phải cho thấy:

- `--variant tddi_paper_member`;
- task-file `outputs/tasks/tail_to_head_tasks.json`;
- output namespace chứa `p3_seed0`;
- member seeds `409845317`, `215626784`, `3041879697`;
- batch 64, effective batch 1024, epochs 20;
- không có `--execute` và không tạo model.

## 5. Biến dùng chung

Chạy lại khi mở terminal mới:

```bash
cd "$HOME/DrugDrug/cil-tddi/cil-tddi"
conda activate ai_env

export STUDY_CONFIG="configs/tddi_ensemble3_replay_distill_p3_seed0.json"
export FULL_ROOT="outputs/full/tddi_ensemble3_replay_distill_p3_seed0_8tasks_v1"
mkdir -p "$FULL_ROOT"
```

## 6. Hàm chạy đúng một member bằng nohup trên CUDA 0

```bash
run_p3_member () {
  MEMBER_ID="$1"

  nohup env \
    STUDY_CONFIG="$STUDY_CONFIG" \
    FULL_ROOT="$FULL_ROOT" \
    MEMBER_ID="$MEMBER_ID" \
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash -c '
set -euo pipefail
cd "$HOME/DrugDrug/cil-tddi/cil-tddi"

nvidia-smi \
  --id=0 \
  --query-gpu=timestamp,index,name,memory.used,memory.free,utilization.gpu,power.draw \
  --format=csv \
  -l 5 >> "$FULL_ROOT/member_${MEMBER_ID}_nvidia_smi.csv" &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null || true; wait $MONITOR_PID 2>/dev/null || true" EXIT

python src/training/tddi_ensemble3_study.py \
  --config "$STUDY_CONFIG" \
  --member-id "$MEMBER_ID" \
  --execute
' >> "$FULL_ROOT/member_${MEMBER_ID}_driver.log" 2>&1 < /dev/null &

  echo $! | tee "$FULL_ROOT/member_${MEMBER_ID}_driver.pid"
}
```

Orchestrator tự quyết định `fresh`, `resume` hoặc `complete`. Không tự tạo trước thư
mục `member_N`; chỉ tạo `FULL_ROOT` để chứa driver log.

## 7. Chạy member 0

```bash
test ! -e "$FULL_ROOT/member_0" || {
  echo "[STOP] member_0 đã tồn tại; kiểm tra PID/checkpoint trước."
  false
}

run_p3_member 0
```

Theo dõi:

```bash
ps -fp "$(cat "$FULL_ROOT/member_0_driver.pid")"
tail -f "$FULL_ROOT/member_0/stdout.log"
nvidia-smi
```

Tìm tiến độ/lỗi:

```bash
grep -E \
  "task_id=|epoch=|latest_replay_distill_state.pt|Traceback|RuntimeError|out of memory" \
  "$FULL_ROOT/member_0/stdout.log" | tail -n 100
```

## 8. Resume một member bị gián đoạn

Ví dụ member 0:

```bash
MEMBER_ID=0
pgrep -afu "$USER" "train_cil.py.*member_${MEMBER_ID}" || true
test -s "$FULL_ROOT/member_${MEMBER_ID}/checkpoints/latest_replay_distill_state.pt"
```

Chỉ khi process cũ đã chết, gọi lại:

```bash
run_p3_member "$MEMBER_ID"
```

Orchestrator validate task hash, backbone, seed, class map, replay budget và config rồi
tự truyền `--resume-replay-checkpoint`. Nếu dừng giữa một task, task đó sẽ chạy lại từ
boundary trước; task hoàn tất không bị export lại.

## 9. Acceptance sau một member

Đặt member cần kiểm tra:

```bash
MEMBER_ID=0
MEMBER_ROOT="$FULL_ROOT/member_${MEMBER_ID}"

test -s "$MEMBER_ROOT/run_summary.md"
test -s "$MEMBER_ROOT/metrics.csv"
test -s "$MEMBER_ROOT/forgetting.csv"
test -s "$MEMBER_ROOT/class_forgetting.csv"
test -s "$MEMBER_ROOT/checkpoints/latest_replay_distill_state.pt"
test -s "$MEMBER_ROOT/member_predictions/task_7/validation.npz"
test -s "$MEMBER_ROOT/member_predictions/task_7/test.npz"
test "$(find "$MEMBER_ROOT/member_predictions" -name '*.npz' | wc -l)" -eq 16
```

Kiểm tra checkpoint/audit:

```bash
python - <<PY
from pathlib import Path
import pandas as pd
import torch

root = Path("$MEMBER_ROOT")
state = torch.load(
    root / "checkpoints/latest_replay_distill_state.pt",
    map_location="cpu",
    weights_only=False,
)
audit = pd.read_csv(root / "training_audit.csv")

print("completed task:", state["completed_task_id"])
print("next task:", state["next_task_id"])
print("seen classes:", len(state["seen_class_map"]))
print("memory rows:", state["replay_buffer_state"]["features"].shape[0])
print(audit[[
    "task", "epochs_trained", "memory_after", "distillation_active",
    "actual_replay_draws_per_epoch", "effective_batch_size",
    "gradient_accumulation_steps"
]].to_string(index=False))

assert state["completed_task_id"] == 7
assert state["next_task_id"] == 8
assert len(state["seen_class_map"]) == 178
assert state["replay_buffer_state"]["features"].shape[0] == 6800
assert audit["task"].astype(int).tolist() == list(range(8))
assert int(audit.iloc[0]["actual_replay_draws_per_epoch"]) == 0
assert (audit.loc[audit["task"] > 0, "actual_replay_draws_per_epoch"] == 6800).all()
assert bool(audit.loc[audit["task"] > 0, "distillation_active"].all())
assert (audit["effective_batch_size"] == 1024).all()
assert (audit["gradient_accumulation_steps"] == 16).all()
print("[PASS] member completed full P3")
PY
```

## 10. Chạy member 1 và member 2

Chỉ chạy member 1 sau khi member 0 pass acceptance:

```bash
run_p3_member 1
```

Sau khi member 1 hoàn tất và pass acceptance, chạy member 2:

```bash
run_p3_member 2
```

Không chạy hai lệnh cùng lúc. Dùng cùng mục theo dõi và acceptance, chỉ đổi
`MEMBER_ID`.

Khi member 2 kết thúc và cả ba member đều hợp lệ, orchestrator có thể tạo luôn các
offline ensemble artifact. Lệnh ở mục 11 vẫn an toàn: nó validate rồi skip artifact
đã hoàn tất, không ghi đè.

## 11. Offline ensemble

Sau khi đủ ba member, gọi orchestrator không giới hạn member. Nó skip ba run hoàn tất
và tạo ensemble validation/test cho task 0–7:

```bash
nohup env PYTHONUNBUFFERED=1 \
  python src/training/tddi_ensemble3_study.py \
  --config "$STUDY_CONFIG" \
  --execute \
  > "$FULL_ROOT/offline_ensemble_driver.log" 2>&1 < /dev/null &

echo $! | tee "$FULL_ROOT/offline_ensemble_driver.pid"
```

```bash
tail -f "$FULL_ROOT/offline_ensemble_driver.log"
test "$(find "$FULL_ROOT/offline_ensemble" -name '*.npz' | wc -l)" -eq 16
test -s "$FULL_ROOT/study_manifest.json"
```

Ensemble luôn mean probabilities và tự fail nếu sample IDs, labels hoặc raw class
column order giữa ba member không khớp.

## 12. UE audit

Tạo train-count artifact một lần:

```bash
mkdir -p "$FULL_ROOT/ue_audit_inputs"

python - <<PY
from pathlib import Path
import pandas as pd
from src.data.ddi_dataset import load_class_counts

path = Path("$FULL_ROOT/ue_audit_inputs/train_class_counts.csv")
if not path.exists():
    counts = load_class_counts("train_extracted.parquet")
    pd.DataFrame(sorted(counts.items()), columns=["class_id", "count"]).to_csv(
        path, index=False
    )
print(path)
PY
```

Chạy UE cho đủ task/split:

```bash
nohup env FULL_ROOT="$FULL_ROOT" bash -c '
set -euo pipefail
cd "$HOME/DrugDrug/cil-tddi/cil-tddi"
TASK_FILE="outputs/tasks/tail_to_head_tasks.json"
COUNTS="$FULL_ROOT/ue_audit_inputs/train_class_counts.csv"

for task in {0..7}; do
  mkdir -p "$FULL_ROOT/ue_audit/task_${task}"
  for split in validation test; do
    JSON_OUT="$FULL_ROOT/ue_audit/task_${task}/${split}.json"
    CSV_OUT="$FULL_ROOT/ue_audit/task_${task}/${split}.csv"
    if [ -s "$JSON_OUT" ] && [ -s "$CSV_OUT" ]; then
      continue
    fi
    test ! -e "$JSON_OUT" && test ! -e "$CSV_OUT"
    python src/eval/offline_ue_audit.py \
      --ensemble "$FULL_ROOT/offline_ensemble/task_${task}/${split}.npz" \
      --task-file "$TASK_FILE" \
      --train-class-counts "$COUNTS" \
      --out-json "$JSON_OUT" \
      --out-csv "$CSV_OUT"
  done
done
' > "$FULL_ROOT/ue_audit_nohup.log" 2>&1 < /dev/null &

echo $! | tee "$FULL_ROOT/ue_audit_nohup.pid"
```

```bash
tail -f "$FULL_ROOT/ue_audit_nohup.log"
test "$(find "$FULL_ROOT/ue_audit" -name '*.json' | wc -l)" -eq 16
test "$(find "$FULL_ROOT/ue_audit" -name '*.csv' | wc -l)" -eq 16
```

## 13. Threshold validation → frozen → test

Primary config:

```text
configs/tddi_ensemble_confidence_threshold_full_p3_primary.json
```

```bash
nohup env FULL_ROOT="$FULL_ROOT" bash -c '
set -euo pipefail
cd "$HOME/DrugDrug/cil-tddi/cil-tddi"
CONFIG="configs/tddi_ensemble_confidence_threshold_full_p3_primary.json"

for task in {0..7}; do
  OUT="$FULL_ROOT/threshold/task_${task}"
  mkdir -p "$OUT"

  if [ ! -s "$OUT/frozen_threshold.json" ]; then
    python src/eval/confidence_threshold.py select \
      --ensemble "$FULL_ROOT/offline_ensemble/task_${task}/validation.npz" \
      --config "$CONFIG" \
      --threshold-out "$OUT/frozen_threshold.json" \
      --report-out "$OUT/validation_report.json"
  fi

  if [ ! -s "$OUT/test_report.json" ]; then
    python src/eval/confidence_threshold.py evaluate \
      --ensemble "$FULL_ROOT/offline_ensemble/task_${task}/test.npz" \
      --config "$CONFIG" \
      --threshold-artifact "$OUT/frozen_threshold.json" \
      --report-out "$OUT/test_report.json"
  fi
done
' > "$FULL_ROOT/threshold_nohup.log" 2>&1 < /dev/null &

echo $! | tee "$FULL_ROOT/threshold_nohup.pid"
```

Không dùng test để chọn threshold. Config low-coverage chỉ dành cho sensitivity
analysis, không thay primary result.

## 14. Temperature calibration tùy chọn

Đây là phân tích bổ sung, không thay đổi raw ensemble hay raw UE. Fit chỉ trên
validation, sau đó test load đúng frozen temperature:

```bash
nohup env FULL_ROOT="$FULL_ROOT" bash -c '
set -euo pipefail
cd "$HOME/DrugDrug/cil-tddi/cil-tddi"
CONFIG="configs/tddi_ensemble_temperature_calibration_full_p3.json"

for task in {0..7}; do
  OUT="$FULL_ROOT/calibration/task_${task}"
  mkdir -p "$OUT"

  if [ ! -s "$OUT/frozen_temperature.json" ]; then
    python src/eval/offline_temperature_calibration.py fit \
      --ensemble "$FULL_ROOT/offline_ensemble/task_${task}/validation.npz" \
      --config "$CONFIG" \
      --temperature-out "$OUT/frozen_temperature.json" \
      --report-out "$OUT/validation_report.json" \
      --calibrated-out "$OUT/validation_calibrated.npz"
  fi

  if [ ! -s "$OUT/test_report.json" ]; then
    python src/eval/offline_temperature_calibration.py evaluate \
      --ensemble "$FULL_ROOT/offline_ensemble/task_${task}/test.npz" \
      --config "$CONFIG" \
      --temperature-artifact "$OUT/frozen_temperature.json" \
      --report-out "$OUT/test_report.json" \
      --calibrated-out "$OUT/test_calibrated.npz"
  fi
done
' > "$FULL_ROOT/calibration_nohup.log" 2>&1 < /dev/null &

echo $! | tee "$FULL_ROOT/calibration_nohup.pid"
```

Primary threshold ở mục 13 vẫn ghi `probability_source=raw`. Không trộn calibrated
probability vào primary report nếu chưa tạo một config threshold riêng ghi rõ nguồn.

## 15. Trình tự ngắn gọn

1. Preflight data/P3/GPU/RAM/disk.
2. Unit tests và dry-run.
3. Member 0 → acceptance.
4. Member 1 → acceptance.
5. Member 2 → acceptance.
6. Offline ensemble.
7. UE audit.
8. Threshold trên validation → frozen artifact → test report.
9. Temperature calibration nếu cần báo cáo calibration phụ.
