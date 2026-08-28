# Runbook: P0–P8 với T-DDI

## 1. Kiểm tra môi trường

```bash
cd /media/neeyu/hien_3/uyen/data_splits
test -x .venv/bin/python
.venv/bin/python -c "import torch, pandas, pyarrow, sklearn; print(torch.__version__, torch.cuda.is_available())"
df -h .
free -h
```

Nếu phải cài lại môi trường:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## 2. Kiểm tra artifact đầu vào

```bash
test -s train_extracted.parquet
test -s validation_extracted.parquet
test -s test_extracted.parquet
test -s outputs/audit/feature_columns.json
test -s outputs/preprocess/scaler.pkl
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
bash -n scripts/run_protocol_study.sh
```

Task files chuẩn phải nằm trong `outputs/tasks/`. Runner sẽ báo lỗi và dừng nếu file cần dùng bị thiếu.

## 3. Tạo lại P0–P4 khi cần

Lệnh này tạo ở thư mục tạm để audit/diff trước khi thay artifact chuẩn.

```bash
protocol_tmpdir="$(mktemp -d)"
.venv/bin/python scripts/build_cil_tasks.py \
  --class-counts outputs/class_distribution/class_counts_train.csv \
  --validation-counts outputs/class_distribution/class_counts_validation.csv \
  --test-counts outputs/class_distribution/class_counts_test.csv \
  --outdir "$protocol_tmpdir" \
  --protocol all \
  --seeds 0 1 2 3 4
```

## 4. Tạo lại P5–P8 khi cần

P5–P8 cần static T-DDI reference và signals. Không dùng test trong bước này. Artifact hiện tại nằm ở `outputs/advanced_protocols/`.

```bash
.venv/bin/python scripts/prepare_advanced_protocol_signals.py \
  --train train_extracted.parquet \
  --validation validation_extracted.parquet \
  --feature-cols outputs/audit/feature_columns.json \
  --scaler outputs/preprocess/scaler.pkl \
  --checkpoint outputs/advanced_protocols/static_tddi_reference/best_model.pt \
  --class-map outputs/advanced_protocols/static_tddi_reference/global_class_map.json \
  --outdir outputs/advanced_protocols/signals_new \
  --variant tddi --batch-size 1024 --device auto
```

Sau khi kiểm tra signals, build task vào thư mục mới để tránh ghi đè:

```bash
.venv/bin/python scripts/build_advanced_protocols.py \
  --class-counts outputs/class_distribution/class_counts_train.csv \
  --class-stats outputs/advanced_protocols/signals/class_protocol_stats.csv \
  --difficulty outputs/advanced_protocols/signals/validation_class_difficulty.csv \
  --confusion-edges outputs/advanced_protocols/signals/validation_confusion_edges.csv \
  --outdir outputs/tasks_new \
  --protocol all --seeds 0 1 2 3 4
```

## 5. Chạy trong tmux

Một protocol:

```bash
tmux new-session -d -s ddi_p4 \
  "cd /media/neeyu/hien_3/uyen/data_splits && bash scripts/run_protocol_study.sh P4 2>&1 | tee outputs/runs_backbones/p4_tddi_driver.log"
```

Tất cả protocol:

```bash
tmux new-session -d -s ddi_protocols \
  "cd /media/neeyu/hien_3/uyen/data_splits && bash scripts/run_protocol_study.sh all 2>&1 | tee outputs/runs_backbones/protocol_study_tddi_driver.log"
```

Kiểm tra session/log:

```bash
tmux ls
tmux attach -t ddi_protocols
tail -f /media/neeyu/hien_3/uyen/data_splits/outputs/runs_backbones/protocol_study_tddi_driver.log
```

Nhấn `Ctrl-b`, sau đó `d` để detach. Không dùng đường dẫn `~/outputs/...`; output nằm dưới root của repo.

## 6. Cơ chế bảo vệ máy

Runner kiểm tra trước mỗi run:

- RAM khả dụng tối thiểu 12 GiB (`PROTOCOL_MIN_AVAILABLE_GIB` có thể override có chủ đích);
- ổ đĩa trống tối thiểu 50 GiB (`PROTOCOL_MIN_DISK_GIB` có thể override);
- task file tồn tại và không rỗng;
- run hoàn tất thì skip;
- run directory tồn tại nhưng chưa hoàn tất thì dừng, không overwrite;
- chạy tuần tự từng protocol/seed, không khởi chạy nhiều GPU job song song.

Ví dụ chỉ khi đã tự kiểm tra máy và cần hạ ngưỡng:

```bash
PROTOCOL_MIN_AVAILABLE_GIB=8 PROTOCOL_MIN_DISK_GIB=30 \
  bash scripts/run_protocol_study.sh P4
```

## 7. Điều kiện một run hoàn tất

Một run chỉ được tính khi cùng tồn tại và không rỗng:

```text
run_summary.md
metrics.csv
forgetting.csv
```

`run_config.json` ghi argument, protocol, seed, git state và checksum implementation để truy vết.

## 8. Smoke test T-DDI Ensemble3 EWC trên máy GPU riêng

Phần này chỉ dành cho study `EWC × tddi_ensemble3 × P4 × seed 0`. Không dùng config
full tám task. Hai phase đều chỉ chạy `member_id=0`:

- Phase A: `configs/tddi_ensemble3_ewc_p4_seed0_smoke_task0.json`, chỉ task 0.
- Phase B: `configs/tddi_ensemble3_ewc_p4_seed0_smoke_tasks01.json`, task 0 rồi task 1;
  chỉ chạy sau khi Phase A được đánh giá `GO`.

Phase B là run mới trong namespace riêng, không resume checkpoint Phase A. Checkpoint
EWC khóa hash task file và số task, nên resume checkpoint một-task vào config hai-task
sẽ bị từ chối đúng thiết kế.

### 8.1. Setup và ghi nhận môi trường

Trên máy GPU, đặt repo và dữ liệu vào cùng workspace rồi chạy từ root repo:

```bash
set -euo pipefail
cd /path/to/DDI-CIL
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p outputs/smoke/environment
git rev-parse HEAD | tee outputs/smoke/environment/git_commit.txt
git status --short | tee outputs/smoke/environment/git_status.txt
python -m pip freeze | tee outputs/smoke/environment/pip_freeze.txt
nvidia-smi | tee outputs/smoke/environment/nvidia_smi.txt
python - <<'PY' | tee outputs/smoke/environment/torch_cuda.txt
import torch
print("torch_version=", torch.__version__)
print("torch_cuda_build=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
print("device_count=", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name=", torch.cuda.get_device_name(0))
    print("device_total_gib=", torch.cuda.get_device_properties(0).total_memory / 2**30)
PY
```

Không chạy smoke nếu `cuda_available=False`. Kiểm tra input và đúng 3.780 feature:

```bash
test -s train_extracted.parquet
test -s validation_extracted.parquet
test -s test_extracted.parquet
test -s outputs/audit/feature_columns.json
test -s outputs/preprocess/scaler.pkl
test -s outputs/tasks/constrained_mass_balanced_seed0_tasks.json

python - <<'PY'
import json
from pathlib import Path

features = json.loads(Path("outputs/audit/feature_columns.json").read_text())
assert len(features) == 3780, len(features)
full = json.loads(Path("outputs/tasks/constrained_mass_balanced_seed0_tasks.json").read_text())
task0 = json.loads(Path("configs/smoke/p4_seed0_task0.json").read_text())
tasks01 = json.loads(Path("configs/smoke/p4_seed0_tasks01.json").read_text())
assert task0["tasks"] == full["tasks"][:1]
assert tasks01["tasks"] == full["tasks"][:2]
print("inputs_ok=true features=3780 task0_classes=38 task01_seen_classes=58")
PY
```

### 8.2. Kiểm tra command, chưa train

Lệnh dưới đây chỉ dry-run. Nó phải hiện `member=0`, seed `409845317`, batch 64,
effective batch 1024 và không tạo member output:

```bash
python src/training/tddi_ensemble3_study.py \
  --config configs/tddi_ensemble3_ewc_p4_seed0_smoke_task0.json \
  --member-id 0
```

Thông số đã khóa trong cả hai smoke config:

```text
microbatch_size                = 64
effective_batch_size           = 1024
gradient_accumulation_steps    = 1024 / 64 = 16
epochs_per_task                = 3
learning_rate                  = 0.001
weight_decay                   = 0.0001
dropout                        = 0.2
ewc_lambda                     = 1000
focal_gamma                    = 1
```

### 8.3. Hàm chạy một smoke phase và đo GPU peak

Paste hàm sau vào shell trên máy GPU. Hàm đọc command trực tiếp từ smoke config, chỉ
chọn member 0, chạy training trong cùng Python process để lấy chính xác PyTorch peak
allocated/reserved VRAM, đồng thời ghi chuỗi `nvidia-smi` mỗi giây.

```bash
run_tddi_smoke () {
  local smoke_config="$1"
  local smoke_root="$2"
  local driver_log="${smoke_root}/driver.log"
  local telemetry="${smoke_root}/resource_telemetry.json"
  local gpu_log="${smoke_root}/nvidia_smi.csv"

  mkdir -p "${smoke_root}"
  export CUDA_VISIBLE_DEVICES=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  nvidia-smi \
    --query-gpu=timestamp,index,name,memory.used,memory.free,utilization.gpu,power.draw \
    --format=csv -l 1 > "${gpu_log}" &
  local monitor_pid=$!

  set +e
  SMOKE_CONFIG="${smoke_config}" SMOKE_TELEMETRY="${telemetry}" \
    .venv/bin/python - <<'PY' 2>&1 | tee "${driver_log}"
import json
import os
import runpy
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from src.training.tddi_ensemble3_study import build_study_plan, load_study_config

config = load_study_config(os.environ["SMOKE_CONFIG"])
plan = build_study_plan(config, python_executable=sys.executable, selected_member_ids=[0])
member = plan.members[0]
if member.command is None:
    raise RuntimeError(f"No runnable member-0 command; status={member.status}")
if member.status not in {"fresh", "resume"}:
    raise RuntimeError(f"Unexpected smoke status: {member.status}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; refusing GPU smoke test")

torch.cuda.set_device(0)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(0)
started = time.perf_counter()
started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
status = "failed"
try:
    sys.argv = [member.command[1], *member.command[2:]]
    runpy.run_path(member.command[1], run_name="__main__")
    status = "completed"
finally:
    try:
        torch.cuda.synchronize(0)
    except Exception:
        pass
    payload = {
        "status": status,
        "config": str(config.source_path),
        "config_sha256": config.sha256,
        "member_id": member.member_id,
        "member_seed": member.member_seed,
        "microbatch_size": config.microbatch_size,
        "effective_batch_size": config.effective_batch_size,
        "gradient_accumulation_steps": config.effective_batch_size // config.microbatch_size,
        "started_at_utc": started_utc,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_runtime_seconds": time.perf_counter() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(0) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(0) / 2**20,
        "device_total_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
    }
    path = Path(os.environ["SMOKE_TELEMETRY"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
PY
  local train_status=${PIPESTATUS[0]}
  set -e
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
  return "${train_status}"
}
```

### 8.4. Phase A — chỉ task 0

Không được dùng `--execute` trên config full tám task. Chạy đúng lệnh:

```bash
run_tddi_smoke \
  configs/tddi_ensemble3_ewc_p4_seed0_smoke_task0.json \
  outputs/smoke/tddi_ensemble3_ewc_p4_seed0_task0
```

Output member nằm tại:

```text
outputs/smoke/tddi_ensemble3_ewc_p4_seed0_task0/member_0/
```

### 8.5. Thu checkpoint, Fisher, head, loss, validation và runtime/epoch

Sau khi phase hoàn tất, đặt `SMOKE_MEMBER_DIR` và `SMOKE_AUDIT_ROOT` tương ứng rồi chạy
audit read-only dưới đây. Với Phase A, head cuối phải có 38 hàng. Với Phase B, head
cuối phải có 58 hàng.

```bash
export SMOKE_MEMBER_DIR=outputs/smoke/tddi_ensemble3_ewc_p4_seed0_task0/member_0
export SMOKE_AUDIT_ROOT=outputs/smoke/tddi_ensemble3_ewc_p4_seed0_task0

.venv/bin/python - <<'PY'
import csv
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path

import torch

member_dir = Path(os.environ["SMOKE_MEMBER_DIR"])
audit_root = Path(os.environ["SMOKE_AUDIT_ROOT"])
checkpoint_path = member_dir / "checkpoints/latest_ewc_state.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
fisher = checkpoint["fisher_total"]

count = zero_count = 0
value_sum = square_sum = 0.0
minimum, maximum = math.inf, -math.inf
all_finite = all_nonnegative = True
for tensor in fisher.values():
    values = tensor.detach().float()
    count += values.numel()
    zero_count += int(torch.count_nonzero(values == 0).item())
    value_sum += float(values.sum().item())
    square_sum += float(torch.linalg.vector_norm(values).item()) ** 2
    minimum = min(minimum, float(values.min().item()))
    maximum = max(maximum, float(values.max().item()))
    all_finite = all_finite and bool(torch.isfinite(values).all())
    all_nonnegative = all_nonnegative and bool((values >= 0).all())

events = []
with (member_dir / "events.csv").open(newline="", encoding="utf-8") as handle:
    events = list(csv.DictReader(handle))
epoch_rows = []
previous_time = None
for row in events:
    if row["event_type"] == "run_started":
        previous_time = datetime.fromisoformat(row["timestamp"])
    if row["event_type"] != "ewc_loss_components":
        continue
    timestamp = datetime.fromisoformat(row["timestamp"])
    payload = json.loads(row["payload_json"])
    message = row["message"]
    for key in ("val_loss", "val_macro_f1", "val_bal_acc"):
        match = re.search(rf"{key}=([-+0-9.eE]+)", message)
        payload[key] = float(match.group(1)) if match else None
    payload["logged_epoch_interval_seconds"] = (
        (timestamp - previous_time).total_seconds() if previous_time else None
    )
    previous_time = timestamp
    epoch_rows.append(payload)

head_weight = checkpoint["model_state"]["head.weight"]
head_bias = checkpoint["model_state"]["head.bias"]
audit = {
    "checkpoint": str(checkpoint_path),
    "checkpoint_size_mib": checkpoint_path.stat().st_size / 2**20,
    "completed_task_id": checkpoint["completed_task_id"],
    "seen_class_count": len(checkpoint["seen_class_map"]),
    "head_weight_shape": list(head_weight.shape),
    "head_bias_shape": list(head_bias.shape),
    "fisher": {
        "parameter_count": count,
        "zero_count": zero_count,
        "nonzero_count": count - zero_count,
        "zero_fraction": zero_count / count,
        "mean": value_sum / count,
        "l2_norm": math.sqrt(square_sum),
        "min": minimum,
        "max": maximum,
        "all_finite": all_finite,
        "all_nonnegative": all_nonnegative,
        "head_weight_shape": list(fisher["head.weight"].shape),
        "head_bias_shape": list(fisher["head.bias"].shape),
    },
    "epochs": epoch_rows,
}
(audit_root / "checkpoint_loss_validation_audit.json").write_text(
    json.dumps(audit, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(audit, indent=2, sort_keys=True))
PY

du -h "${SMOKE_MEMBER_DIR}/checkpoints/latest_ewc_state.pt"
cat "${SMOKE_AUDIT_ROOT}/resource_telemetry.json"
cat "${SMOKE_AUDIT_ROOT}/checkpoint_loss_validation_audit.json"
cat "${SMOKE_MEMBER_DIR}/run_summary.md"
```

`logged_epoch_interval_seconds` là wall-clock giữa hai event epoch liên tiếp; epoch đầu
còn bao gồm setup/data loading ban đầu. `resource_telemetry.json` là wall-clock toàn
phase và peak VRAM chính xác theo PyTorch allocator. `nvidia_smi.csv` là kiểm tra chéo
mức dùng toàn GPU.

### 8.6. Tiêu chí GO sau task 0

Chỉ chuyển Phase B khi tất cả điều kiện sau đúng:

- process kết thúc code 0, không có OOM, traceback, NaN hoặc Inf;
- `run_summary.md`, `metrics.csv`, `forgetting.csv` và
  `checkpoints/latest_ewc_state.pt` tồn tại, không rỗng;
- `completed_task_id=0`, `seen_class_count=38`, head weight/bias có 38 hàng;
- Fisher finite, không âm và có ít nhất một phần tử khác 0;
- ba epoch có classification/total loss finite; ở task 0 raw/scaled EWC penalty bằng
  0 là đúng vì chưa có task cũ;
- validation loss, Macro-F1 và balanced accuracy đều finite;
- peak reserved VRAM còn headroom an toàn, khuyến nghị không vượt 90% tổng VRAM;
- runtime/epoch và checkpoint size phù hợp giới hạn vận hành của máy GPU.

Nếu một điều kiện không đạt, kết luận `NO-GO` và chưa chạy task 1/full study.

### 8.7. Phase B — task 0 rồi task 1, chỉ sau GO

Sau quyết định GO có chủ đích, dry-run trước:

```bash
python src/training/tddi_ensemble3_study.py \
  --config configs/tddi_ensemble3_ewc_p4_seed0_smoke_tasks01.json \
  --member-id 0
```

Sau đó mới chạy:

```bash
run_tddi_smoke \
  configs/tddi_ensemble3_ewc_p4_seed0_smoke_tasks01.json \
  outputs/smoke/tddi_ensemble3_ewc_p4_seed0_tasks01
```

Audit lại với:

```bash
export SMOKE_MEMBER_DIR=outputs/smoke/tddi_ensemble3_ewc_p4_seed0_tasks01/member_0
export SMOKE_AUDIT_ROOT=outputs/smoke/tddi_ensemble3_ewc_p4_seed0_tasks01
# Chạy lại block audit ở mục 8.5.
```

Task 1 đạt kỹ thuật khi checkpoint có `completed_task_id=1`, head có 58 hàng, Fisher
finite/nonnegative, và các loss component task 1 finite. Từ task 1,
`raw_ewc_penalty`/`scaled_ewc_penalty` phải xuất hiện và không âm.

### 8.8. Xử lý OOM không đổi effective batch/hyperparameter

Không xóa hoặc ghi đè output OOM. Giữ learning rate, dropout, EWC lambda, focal gamma
và `effective_batch_size=1024`. Tạo config retry trong namespace mới, giảm microbatch
64 xuống 32 và tăng accumulation 16 lên 32:

```bash
python - <<'PY'
import json
from pathlib import Path

source = Path("configs/tddi_ensemble3_ewc_p4_seed0_smoke_task0.json")
target = Path("outputs/smoke/local_task0_mb32_retry1.json")
payload = json.loads(source.read_text())
payload["training"]["microbatch_size"] = 32
payload["training"]["effective_batch_size"] = 1024
payload["training"]["gradient_accumulation_steps"] = 32
payload["outputs"]["root"] = "outputs/smoke/tddi_ensemble3_ewc_p4_seed0_task0_mb32_retry1"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(payload, indent=2) + "\n")
print(target)
PY

python src/training/tddi_ensemble3_study.py \
  --config outputs/smoke/local_task0_mb32_retry1.json --member-id 0
```

Kiểm tra dry-run phải hiện `--batch-size 32 --effective-batch-size 1024`. Chỉ sau đó
gọi `run_tddi_smoke` với config retry và output root mới. Không giảm effective batch,
không đổi hyperparameter âm thầm, và không reuse directory của lần OOM.

### 8.9. Artifact cần gửi lại để quyết định GO/NO-GO

Nén và gửi các file nhỏ sau; không cần gửi checkpoint model lớn trừ khi được yêu cầu:

```text
outputs/smoke/environment/git_commit.txt
outputs/smoke/environment/git_status.txt
outputs/smoke/environment/pip_freeze.txt
outputs/smoke/environment/nvidia_smi.txt
outputs/smoke/environment/torch_cuda.txt
<smoke_root>/driver.log
<smoke_root>/nvidia_smi.csv
<smoke_root>/resource_telemetry.json
<smoke_root>/checkpoint_loss_validation_audit.json
<member_0>/run_config.json
<member_0>/train.log
<member_0>/events.csv
<member_0>/training_audit.csv
<member_0>/run_summary.md
<member_0>/metrics.csv
<member_0>/forgetting.csv
<member_0>/seen_class_map_task_0.json
<member_0>/seen_class_map_task_1.json          # chỉ Phase B
```

Ngoài ra gửi checksum và kích thước checkpoint thay vì upload checkpoint:

```bash
sha256sum "${SMOKE_MEMBER_DIR}/checkpoints/latest_ewc_state.pt" \
  | tee "${SMOKE_AUDIT_ROOT}/checkpoint_sha256.txt"
du -h "${SMOKE_MEMBER_DIR}/checkpoints/latest_ewc_state.pt" \
  | tee "${SMOKE_AUDIT_ROOT}/checkpoint_size.txt"
```

Không có command nào trong mục này tự động chuyển sang full tám task hoặc member 1/2.
