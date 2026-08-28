# Runbook chi tiết: train T-DDI Ensemble3 EWC trên máy GPU khác

## 1. Mục tiêu và nguyên tắc an toàn

Runbook này dành cho study:

```text
method          = EWC
backbone        = tddi_ensemble3 (ba tddi_paper_member độc lập)
protocol        = P4
experiment seed = 0
member IDs      = 0, 1, 2
```

Trình tự bắt buộc:

```text
kiểm tra máy
  -> chạy unit tests và CUDA sanity test
  -> dry-run command
  -> smoke member 0 / task 0
  -> đánh giá GO/NO-GO
  -> smoke member 0 / task 0+1
  -> đánh giá GO/NO-GO lần hai
  -> mới chạy full member 0 -> member 1 -> member 2
  -> offline ensemble
```

Không chạy ba member đồng thời. Không xóa output lỗi và không reuse một directory đã
có dữ liệu nhưng không có checkpoint hợp lệ. Mặc định orchestrator chỉ dry-run; chỉ
`--execute` mới bắt đầu training.

## 2. Các file cần chuyển sang máy GPU

Chuyển toàn bộ repository, tối thiểu phải có:

```text
train_extracted.parquet
validation_extracted.parquet
test_extracted.parquet
outputs/audit/feature_columns.json
outputs/preprocess/scaler.pkl
outputs/tasks/constrained_mass_balanced_seed0_tasks.json
configs/tddi_ensemble3_ewc_p4_seed0.json
configs/tddi_ensemble3_ewc_p4_seed0_smoke_task0.json
configs/tddi_ensemble3_ewc_p4_seed0_smoke_tasks01.json
configs/smoke/p4_seed0_task0.json
configs/smoke/p4_seed0_tasks01.json
requirements.txt
src/
tests/
```

Nên dùng `rsync` để có resume và checksum trong lúc chuyển dữ liệu:

```bash
rsync -avh --partial --progress \
  /local/path/DDI-CIL/ user@gpu-host:/remote/path/DDI-CIL/
```

Sau khi chuyển, so sánh commit và checksum các input lớn:

```bash
cd /remote/path/DDI-CIL
git rev-parse HEAD
sha256sum train_extracted.parquet validation_extracted.parquet test_extracted.parquet \
  | tee transferred_data_sha256.txt
```

## 3. Công thức ước lượng tài nguyên trước khi chạy

### 3.1. Số parameter

Với `C` lớp đã thấy, số parameter của một member là:

```text
P(C) = LayerNorm + Linear1 + Linear2 + head
     = 2×3780
       + (3780×7560 + 7560)
       + (7560×7560 + 7560)
       + (7560×C + C)
     = 85,753,080 + 7,561×C
```

Một số mốc:

| Giai đoạn | C | Parameter | FP32 model | Model + Fisher checkpoint |
|---|---:|---:|---:|---:|
| Task 0 | 38 | 86,040,398 | khoảng 328 MiB | khoảng 656 MiB |
| Task 0+1 | 58 | 86,191,618 | khoảng 329 MiB | khoảng 658 MiB |
| Task cuối | 178 | 87,098,938 | khoảng 332 MiB | khoảng 665 MiB |

Công thức dung lượng tensor FP32:

```text
MiB = số_phần_tử × 4 / 2^20
```

Checkpoint EWC lưu model và Fisher nên phần tensor chính xấp xỉ:

```text
checkpoint_MiB ≈ P(C) × 8 / 2^20
```

File thật có thêm class map, RNG và progress metadata nên sẽ lớn hơn một ít.

### 3.2. VRAM

Ước lượng nền cho một member FP32 khi EWC đang hoạt động:

```text
model weights        ≈ 4P bytes
gradients            ≈ 4P bytes
AdamW moments        ≈ 8P bytes
Fisher               ≈ 4P bytes
theta_star           ≈ 4P bytes
------------------------------------------------
tensor nền           ≈ 24P bytes
```

Ở `C=178`, phần tensor nền khoảng `1.95 GiB`. Đây không phải peak cuối cùng vì PyTorch
còn activation, temporary optimizer buffers, CUDA context, kernel workspace, cache và
thời điểm mở rộng head. Vì vậy phải đo smoke, không được dùng con số 1.95 GiB làm kết
luận máy chắc chắn đủ.

Điều kiện khuyến nghị:

- **GO tốt:** GPU từ 16 GiB, VRAM trống trước run từ 14 GiB.
- **GO ưu tiên:** GPU 24 GiB và VRAM trống từ 20 GiB.
- **Có điều kiện:** GPU 12–16 GiB; phải smoke microbatch 64, có thể giảm microbatch.
- **NO-GO ban đầu:** dưới 12 GiB hoặc GPU đang được process khác sử dụng đáng kể.

Quy tắc headroom sau smoke:

```text
VRAM_headroom = total_VRAM - peak_reserved_VRAM
reserved_ratio = peak_reserved_VRAM / total_VRAM
```

Nên đạt:

```text
reserved_ratio <= 0.90
VRAM_headroom  >= 2 GiB
```

### 3.3. RAM cho feature 3.780 chiều

Raw FP32 feature matrix có kích thước:

```text
feature_GiB = số_dòng × 3780 × 4 / 2^30
```

Trong preprocessing có thời điểm dữ liệu là float64 và tồn tại cả pandas/Arrow/numpy
copy. Dùng hệ số an toàn 2.5–3 lần:

```text
RAM_task_recommended ≈ 3 × feature_GiB + 8 GiB hệ thống/model/checkpoint
```

Khuyến nghị thực tế:

- RAM tổng từ 32 GiB và RAM available từ 24 GiB: phù hợp.
- RAM tổng 16–32 GiB: chỉ GO khi công thức theo row count vẫn còn ít nhất 6–8 GiB dư.
- RAM available dưới 12 GiB: NO-GO cho smoke full-data.

### 3.4. Disk

Upper bound đơn giản cho model/checkpoint của full study:

```text
disk_models ≈ members × (tasks × model_state + latest_EWC_state)
            ≈ 3 × (8 × 332 MiB + 665 MiB)
            ≈ 9.7 GiB
```

Prediction, ensemble, CSV/log và filesystem overhead chưa nằm trong 9.7 GiB. Yêu cầu:

- free disk tối thiểu 30 GiB nếu chỉ smoke;
- free disk tối thiểu 50 GiB trước full ba-member study;
- nên có 70 GiB nếu muốn giữ retry/OOM artifacts.

### 3.5. Gradient accumulation

Công thức:

```text
gradient_accumulation_steps = effective_batch_size / microbatch_size
```

Config chuẩn:

```text
microbatch 64, effective batch 1024 -> accumulation 16
```

Nếu OOM:

```text
microbatch 32, effective batch 1024 -> accumulation 32
microbatch 16, effective batch 1024 -> accumulation 64
```

Effective batch, learning rate, dropout, EWC lambda và focal gamma không được thay đổi
âm thầm khi xử lý OOM.

## 4. Setup môi trường trên máy GPU

Các lệnh dưới đây giả định Linux và đang đứng tại root repo:

```bash
set -euo pipefail
cd /remote/path/DDI-CIL

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

mkdir -p outputs/remote_preflight
git rev-parse HEAD | tee outputs/remote_preflight/git_commit.txt
git status --short | tee outputs/remote_preflight/git_status.txt
python -m pip freeze | tee outputs/remote_preflight/pip_freeze.txt
uname -a | tee outputs/remote_preflight/uname.txt
lscpu | tee outputs/remote_preflight/lscpu.txt
nvidia-smi | tee outputs/remote_preflight/nvidia_smi.txt
```

Nếu `torch.cuda.is_available()` là false, dừng lại và cài PyTorch CUDA build tương thích
driver của máy trước khi tiếp tục.

## 5. Preflight tự động: PASS/WARN/FAIL

Chạy script read-only sau. Nó không train model và không tạo CUDA tensor lớn:

```bash
python - <<'PY' | tee outputs/remote_preflight/preflight_report.txt
import json
import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import torch

required = [
    Path("train_extracted.parquet"),
    Path("validation_extracted.parquet"),
    Path("test_extracted.parquet"),
    Path("outputs/audit/feature_columns.json"),
    Path("outputs/preprocess/scaler.pkl"),
    Path("outputs/tasks/constrained_mass_balanced_seed0_tasks.json"),
]
failures = []
warnings = []

for path in required:
    if not path.is_file() or path.stat().st_size == 0:
        failures.append(f"missing_or_empty:{path}")

features = []
feature_path = Path("outputs/audit/feature_columns.json")
if feature_path.is_file():
    features = json.loads(feature_path.read_text())
    if len(features) != 3780:
        failures.append(f"feature_count={len(features)} expected=3780")

task_path = Path("outputs/tasks/constrained_mass_balanced_seed0_tasks.json")
if task_path.is_file():
    task_spec = json.loads(task_path.read_text())
    if task_spec.get("protocol") != "constrained_mass_balanced":
        failures.append("wrong_protocol")
    if int(task_spec.get("seed", -1)) != 0:
        failures.append("wrong_experiment_seed")
    if len(task_spec.get("tasks", [])) != 8:
        failures.append("task_count_not_8")

row_counts = {}
raw_feature_gib = {}
for split in ("train", "validation", "test"):
    path = Path(f"{split}_extracted.parquet")
    if path.is_file():
        rows = pq.ParquetFile(path).metadata.num_rows
        row_counts[split] = rows
        raw_feature_gib[split] = rows * 3780 * 4 / 2**30

if os.name == "posix" and Path("/proc/meminfo").is_file():
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        fields[key] = int(value.strip().split()[0])
    ram_total_gib = fields["MemTotal"] / 1024**2
    ram_available_gib = fields["MemAvailable"] / 1024**2
else:
    ram_total_gib = ram_available_gib = 0.0

disk = shutil.disk_usage(Path.cwd())
disk_free_gib = disk.free / 2**30
cpu_count = os.cpu_count() or 0

if ram_available_gib and ram_available_gib < 12:
    failures.append(f"available_ram_gib={ram_available_gib:.1f}<12")
elif ram_available_gib and ram_available_gib < 24:
    warnings.append(f"available_ram_gib={ram_available_gib:.1f}<24")
if disk_free_gib < 30:
    failures.append(f"disk_free_gib={disk_free_gib:.1f}<30")
elif disk_free_gib < 50:
    warnings.append(f"disk_free_gib={disk_free_gib:.1f}<50_for_full_study")
if cpu_count < 8:
    warnings.append(f"cpu_threads={cpu_count}<8")

cuda_available = torch.cuda.is_available()
gpu = None
if not cuda_available:
    failures.append("torch_cuda_unavailable")
else:
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    gpu = {
        "name": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "total_gib": total_bytes / 2**30,
        "free_gib": free_bytes / 2**30,
        "torch_cuda_build": torch.version.cuda,
    }
    if gpu["total_gib"] < 12:
        failures.append(f"gpu_total_gib={gpu['total_gib']:.1f}<12")
    elif gpu["total_gib"] < 16:
        warnings.append(f"gpu_total_gib={gpu['total_gib']:.1f}<16")
    if gpu["free_gib"] < 10:
        failures.append(f"gpu_free_gib={gpu['free_gib']:.1f}<10")
    elif gpu["free_gib"] < 14:
        warnings.append(f"gpu_free_gib={gpu['free_gib']:.1f}<14")

report = {
    "status": "FAIL" if failures else ("WARN" if warnings else "PASS"),
    "failures": failures,
    "warnings": warnings,
    "cpu_threads": cpu_count,
    "ram_total_gib": ram_total_gib,
    "ram_available_gib": ram_available_gib,
    "disk_free_gib": disk_free_gib,
    "row_counts": row_counts,
    "raw_fp32_feature_gib": raw_feature_gib,
    "gpu": gpu,
}
print(json.dumps(report, indent=2, sort_keys=True))
if failures:
    raise SystemExit(1)
PY
```

Diễn giải:

- `PASS`: có thể chạy test code và smoke task 0.
- `WARN`: đọc warning, đóng process khác hoặc chuẩn bị fallback microbatch trước.
- `FAIL`: không bắt đầu smoke full-data.

## 6. Test CUDA ngắn trước khi test repo

### 6.1. Kiểm tra allocation và backward

Lệnh này chỉ kiểm tra CUDA/PyTorch/driver, không train DDI:

```bash
python - <<'PY' | tee outputs/remote_preflight/cuda_sanity.txt
import time
import torch

assert torch.cuda.is_available()
device = "cuda:0"
torch.cuda.reset_peak_memory_stats(0)
x = torch.randn(2048, 2048, device=device, requires_grad=True)
started = time.perf_counter()
loss = (x @ x).square().mean()
loss.backward()
torch.cuda.synchronize()
print("loss=", float(loss))
print("seconds=", time.perf_counter() - started)
print("peak_allocated_mib=", torch.cuda.max_memory_allocated(0) / 2**20)
print("peak_reserved_mib=", torch.cuda.max_memory_reserved(0) / 2**20)
assert torch.isfinite(loss)
PY
```

GO nếu command kết thúc code 0, loss finite và không có CUDA/driver error.

### 6.2. Kiểm tra layer có kích thước gần model thật

```bash
python - <<'PY' | tee outputs/remote_preflight/tddi_layer_sanity.txt
import torch
import torch.nn.functional as F

device = "cuda:0"
batch = 64
x = torch.randn(batch, 3780, device=device, requires_grad=True)
layer = torch.nn.Linear(3780, 7560, device=device)
y = F.gelu(layer(x)).square().mean()
y.backward()
torch.cuda.synchronize()
print("output_shape=", tuple(layer(x).shape))
print("loss=", float(y.detach()))
print("peak_allocated_mib=", torch.cuda.max_memory_allocated(0) / 2**20)
assert torch.isfinite(y)
PY
```

## 7. Chạy test code trước training

Chạy nhóm test trực tiếp liên quan study:

```bash
python -m pytest \
  tests/test_ensemble_seed.py \
  tests/test_tddi_paper_member.py \
  tests/test_tddi_paper_member_ewc.py \
  tests/test_ewc_checkpoint.py \
  tests/test_member_prediction_export.py \
  tests/test_offline_ensemble.py \
  tests/test_confidence_threshold.py \
  tests/test_tddi_ensemble3_study.py \
  -q 2>&1 | tee outputs/remote_preflight/study_tests.log
```

Sau đó chạy toàn repo:

```bash
python -m pytest -q 2>&1 | tee outputs/remote_preflight/all_tests.log
```

Chỉ GO nếu toàn bộ test study pass. Nếu full suite fail do dependency, sửa môi trường;
không tự ý skip test rồi chạy full training.

## 8. Xác minh config và command bằng dry-run

Dry-run không train và không tạo member output:

```bash
python src/training/tddi_ensemble3_study.py \
  --config configs/tddi_ensemble3_ewc_p4_seed0_smoke_task0.json \
  --member-id 0 \
  | tee outputs/remote_preflight/task0_dry_run.txt
```

Phải nhìn thấy:

```text
member=0
member_seed=409845317
--method ewc
--variant tddi_paper_member
--batch-size 64
--effective-batch-size 1024
--epochs 3
--seed 0
--member-id 0
```

Nếu command có task file/output khác config smoke, dừng lại.

## 9. Chạy smoke task 0 bằng nohup

Hàm `run_tddi_smoke` có đo PyTorch peak VRAM đã được cung cấp trong
`docs/RUNBOOK.md`, mục 8.3. Paste hàm đó vào shell, sau đó export để `nohup` dùng được:

```bash
source .venv/bin/activate
export -f run_tddi_smoke
mkdir -p outputs/smoke/tddi_ensemble3_ewc_p4_seed0_task0

nohup bash -c '
  cd /remote/path/DDI-CIL
  run_tddi_smoke \
    configs/tddi_ensemble3_ewc_p4_seed0_smoke_task0.json \
    outputs/smoke/tddi_ensemble3_ewc_p4_seed0_task0
' > outputs/smoke/tddi_ensemble3_ewc_p4_seed0_task0/nohup.log 2>&1 &

echo $! | tee outputs/smoke/tddi_ensemble3_ewc_p4_seed0_task0/nohup.pid
```

Theo dõi:

```bash
tail -f outputs/smoke/tddi_ensemble3_ewc_p4_seed0_task0/nohup.log
nvidia-smi
ps -fp "$(cat outputs/smoke/tddi_ensemble3_ewc_p4_seed0_task0/nohup.pid)"
```

Nếu shell không cho export function, chạy foreground bằng lệnh `run_tddi_smoke ...`
trong `tmux`, hoặc dùng command đơn giản sau. Command đơn giản vẫn ghi `nvidia-smi`
nhưng không có PyTorch allocator telemetry chi tiết:

```bash
nohup bash -c '
  cd /remote/path/DDI-CIL
  source .venv/bin/activate
  python src/training/tddi_ensemble3_study.py \
    --config configs/tddi_ensemble3_ewc_p4_seed0_smoke_task0.json \
    --execute --member-id 0
' > outputs/smoke/tddi_ensemble3_ewc_p4_seed0_task0/orchestrator_nohup.log 2>&1 &
```

Không chạy thêm member hoặc full config trong khi smoke còn hoạt động.

## 10. Công thức đánh giá smoke task 0

Sau khi chạy block audit ở `docs/RUNBOOK.md` mục 8.5, tính:

```text
peak_ratio          = peak_reserved_mib / device_total_mib
VRAM_headroom_MiB   = device_total_mib - peak_reserved_mib
mean_epoch_seconds  = trung bình logged_epoch_interval_seconds
checkpoint_GiB      = checkpoint_size_mib / 1024
Fisher_nonzero_rate = nonzero_count / parameter_count
```

Task 0 là `GO` khi:

- exit code 0, không OOM/traceback/NaN/Inf;
- head có 38 hàng và `completed_task_id=0`;
- Fisher finite, không âm, `Fisher_nonzero_rate > 0`;
- classification loss và total loss finite;
- raw/scaled EWC penalty bằng 0 ở task 0 là đúng;
- validation loss, Macro-F1, balanced accuracy finite;
- `peak_ratio <= 0.90` và headroom ít nhất 2 GiB;
- checkpoint, prediction, metrics và log đều được ghi đầy đủ.

Không đặt ngưỡng accuracy tùy ý sau khi nhìn test. Smoke gate này kiểm tra kỹ thuật và
tài nguyên; quyết định chất lượng model sau này phải dựa trên validation protocol đã
định trước.

## 11. Ước lượng thời gian full study từ smoke

Nếu task 0 có `N0` train rows, ba epoch smoke mất `T0` giây, ước lượng thô một member:

```text
T_member ≈ Σ_t [T0 × (N_t/N0) × (E_full/E_smoke)] + Fisher + evaluation
```

Với `E_full=20`, `E_smoke=3`:

```text
epoch_scale = 20/3 ≈ 6.67
T_three_members ≈ 3 × T_member
```

Do validation/test all-seen tăng theo task và mỗi task có thêm Fisher pass, cộng buffer
an toàn 25–40%:

```text
T_planned ≈ 1.25 đến 1.40 × T_three_members
```

Đây chỉ là dự báo để đặt thời gian `nohup`; runtime task 0+1 là ước lượng tốt hơn vì đã
bao gồm head expansion và EWC penalty.

## 12. Smoke task 0+1 bằng nohup, chỉ sau GO

Dry-run trước:

```bash
python src/training/tddi_ensemble3_study.py \
  --config configs/tddi_ensemble3_ewc_p4_seed0_smoke_tasks01.json \
  --member-id 0 \
  | tee outputs/remote_preflight/tasks01_dry_run.txt
```

Sau quyết định GO rõ ràng:

```bash
export -f run_tddi_smoke
mkdir -p outputs/smoke/tddi_ensemble3_ewc_p4_seed0_tasks01

nohup bash -c '
  cd /remote/path/DDI-CIL
  run_tddi_smoke \
    configs/tddi_ensemble3_ewc_p4_seed0_smoke_tasks01.json \
    outputs/smoke/tddi_ensemble3_ewc_p4_seed0_tasks01
' > outputs/smoke/tddi_ensemble3_ewc_p4_seed0_tasks01/nohup.log 2>&1 &

echo $! | tee outputs/smoke/tddi_ensemble3_ewc_p4_seed0_tasks01/nohup.pid
```

Task 0+1 là `GO` khi các điều kiện task 0 vẫn đúng, đồng thời:

- `completed_task_id=1`;
- head có 58 hàng;
- class map chứa 58 lớp đúng P4;
- task 1 có raw/scaled EWC penalty finite và không âm;
- Fisher sau task 1 finite, không âm;
- old/new-class validation không có NaN/Inf;
- peak VRAM vẫn có headroom theo công thức.

## 13. Xử lý OOM

Không xóa output OOM. Dùng hướng dẫn tạo retry config ở `docs/RUNBOOK.md` mục 8.8.
Mỗi retry phải:

- có config JSON riêng;
- có output root riêng;
- giữ effective batch 1024;
- ghi rõ microbatch và accumulation mới;
- không thay learning rate hoặc EWC lambda.

Thứ tự thử:

```text
64 × accumulation 16
32 × accumulation 32
16 × accumulation 64
```

Nếu microbatch 16 vẫn OOM, kết luận máy/config FP32 hiện tại `NO-GO`; không tự bật AMP
hoặc offload vì đó là một thay đổi study cần được triển khai và kiểm chứng riêng.

## 14. Full train sau hai lần GO

Full config:

```text
configs/tddi_ensemble3_ewc_p4_seed0.json
```

Dry-run full một lần nhưng chưa execute:

```bash
python src/training/tddi_ensemble3_study.py \
  --config configs/tddi_ensemble3_ewc_p4_seed0.json \
  | tee outputs/remote_preflight/full_dry_run.txt
```

Tạo output root và chạy từng member riêng. Không chạy ba command cùng lúc.

### Member 0

```bash
mkdir -p outputs/tddi_ensemble3_ewc_p4_seed0
nohup bash -c '
  cd /remote/path/DDI-CIL
  source .venv/bin/activate
  python src/training/tddi_ensemble3_study.py \
    --config configs/tddi_ensemble3_ewc_p4_seed0.json \
    --execute --member-id 0
' > outputs/tddi_ensemble3_ewc_p4_seed0/member0_driver.log 2>&1 &
echo $! | tee outputs/tddi_ensemble3_ewc_p4_seed0/member0_driver.pid
```

### Member 1, chỉ sau khi member 0 hoàn tất

```bash
nohup bash -c '
  cd /remote/path/DDI-CIL
  source .venv/bin/activate
  python src/training/tddi_ensemble3_study.py \
    --config configs/tddi_ensemble3_ewc_p4_seed0.json \
    --execute --member-id 1
' > outputs/tddi_ensemble3_ewc_p4_seed0/member1_driver.log 2>&1 &
echo $! | tee outputs/tddi_ensemble3_ewc_p4_seed0/member1_driver.pid
```

### Member 2, chỉ sau khi member 1 hoàn tất

```bash
nohup bash -c '
  cd /remote/path/DDI-CIL
  source .venv/bin/activate
  python src/training/tddi_ensemble3_study.py \
    --config configs/tddi_ensemble3_ewc_p4_seed0.json \
    --execute --member-id 2
' > outputs/tddi_ensemble3_ewc_p4_seed0/member2_driver.log 2>&1 &
echo $! | tee outputs/tddi_ensemble3_ewc_p4_seed0/member2_driver.pid
```

Sau member 2, orchestrator kiểm tra alignment, tạo offline ensemble cho từng task/split
và ghi `study_manifest.json`.

## 15. Resume sau mất kết nối hoặc reboot

Chạy lại đúng command `--execute --member-id N`. Nếu member có
`checkpoints/latest_ewc_state.pt`, orchestrator tự thêm `--resume-ewc-checkpoint` và
tiếp tục task kế tiếp. Nếu run đã hoàn tất, nó skip và không ghi đè.

Kiểm tra trước resume:

```bash
ls -lh outputs/tddi_ensemble3_ewc_p4_seed0/member_0/checkpoints/latest_ewc_state.pt
tail -n 30 outputs/tddi_ensemble3_ewc_p4_seed0/member_0/train.log
```

Không đổi config, task file, seed, batch hoặc input path khi resume; checkpoint validation
sẽ từ chối metadata/hash khác nhau.

## 16. Theo dõi và dừng process

```bash
tail -f outputs/tddi_ensemble3_ewc_p4_seed0/member0_driver.log
watch -n 1 nvidia-smi
ps -fp "$(cat outputs/tddi_ensemble3_ewc_p4_seed0/member0_driver.pid)"
```

Dừng có kiểm soát:

```bash
kill -TERM "$(cat outputs/tddi_ensemble3_ewc_p4_seed0/member0_driver.pid)"
```

Checkpoint chỉ chắc chắn nhất tại task boundary. Dừng giữa task có thể phải train lại
task đang chạy, nhưng không mất các task đã checkpoint trước đó.

## 17. Checklist cuối cùng trước full train

Chỉ chạy full khi có thể đánh dấu tất cả:

- [ ] Repo/commit và dữ liệu đã đồng bộ checksum.
- [ ] CUDA sanity và representative-layer sanity pass.
- [ ] Tất cả test study pass.
- [ ] Preflight không có FAIL.
- [ ] Có ít nhất 50 GiB disk trống.
- [ ] Có ít nhất 24 GiB system RAM available hoặc công thức row-count chứng minh đủ.
- [ ] GPU có ít nhất 14 GiB free trước run.
- [ ] Smoke task 0 GO.
- [ ] Smoke task 0+1 GO.
- [ ] Peak reserved ratio không vượt 90%.
- [ ] Fisher/head/loss/validation audit finite và đúng shape.
- [ ] Đã ước lượng tổng runtime ba member và có thời gian vận hành phù hợp.
- [ ] Chỉ một member được chạy tại một thời điểm.

## 18. Artifact cần gửi lại khi cần đánh giá

Trước full run, gửi các artifact nhỏ được liệt kê trong `docs/RUNBOOK.md` mục 8.9.
Quan trọng nhất:

```text
preflight_report.txt
cuda_sanity.txt
study_tests.log
resource_telemetry.json
nvidia_smi.csv
checkpoint_loss_validation_audit.json
run_config.json
train.log
events.csv
training_audit.csv
run_summary.md
metrics.csv
forgetting.csv
checkpoint_sha256.txt
checkpoint_size.txt
```

Không cần gửi checkpoint hàng trăm MiB nếu chỉ đánh giá GO/NO-GO; checksum, size và
audit JSON thường là đủ.
